from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import io
import json
import os
import threading
import time
from typing import BinaryIO

import numpy
import onnxruntime
import redis
from PIL import Image, ImageOps

from inference.assets import LABELS_PATH, MODEL_PATH, ensure_model_assets


# Cap concurrent ONNX inference calls. Default 4 — tune based on task CPU allocation.
# dev/staging: 256 CPU units (0.25 vCPU) → 1 concurrent call.
# prod: 512 CPU units (0.5 vCPU) → 4 concurrent calls.
_MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "4"))
_inference_semaphore = threading.Semaphore(_MAX_CONCURRENT_INFERENCES)

# Distributed inference cache. REDIS_URL is optional — caching is skipped if unset.
_INFERENCE_CACHE_TTL = int(os.getenv("INFERENCE_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))
_redis_url = os.getenv("REDIS_URL")
_cache: redis.Redis | None = redis.Redis.from_url(_redis_url) if _redis_url else None

IMAGE_INPUT_SIZE = (224, 224)
IMAGE_MEAN = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
IMAGE_STD = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)


@dataclass(frozen=True)
class TagPrediction:
    label: str
    score: float


@dataclass(frozen=True)
class InferenceResult:
    model_name: str
    tags: list[TagPrediction]
    width: int
    height: int
    preprocessing_ms: float
    model_ms: float
    cache_hit: bool = False


def _serialize_result(result: InferenceResult) -> bytes:
    d = asdict(result)
    d["cache_hit"] = False  # never persist cache_hit=True
    return json.dumps(d).encode()


def _deserialize_result(data: bytes) -> InferenceResult:
    d = json.loads(data)
    return InferenceResult(
        model_name=d["model_name"],
        tags=[TagPrediction(**t) for t in d["tags"]],
        width=d["width"],
        height=d["height"],
        preprocessing_ms=d["preprocessing_ms"],
        model_ms=d["model_ms"],
        cache_hit=True,
    )


class PretrainedImageClassifier:
    def __init__(self) -> None:
        ensure_model_assets()
        self.model_name = MODEL_PATH.name
        self.labels = tuple(
            label.strip()
            for label in LABELS_PATH.read_text().splitlines()
            if label.strip()
        )
        self.session = onnxruntime.InferenceSession(
            MODEL_PATH.as_posix(),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def classify(self, uploaded_image: BinaryIO) -> InferenceResult:
        uploaded_image.seek(0)
        image_bytes = uploaded_image.read()
        cache_key = hashlib.sha256(image_bytes).hexdigest()

        if _cache is not None:
            cached = _cache.get(cache_key)
            if cached is not None:
                return _deserialize_result(cached)

        with Image.open(io.BytesIO(image_bytes)) as image:
            normalized_image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = normalized_image.size

            t0 = time.perf_counter()
            input_tensor = self._build_input_tensor(normalized_image)
            preprocessing_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        with _inference_semaphore:
            raw_output = self.session.run(None, {self.input_name: input_tensor})[0]
        model_ms = (time.perf_counter() - t0) * 1000

        scores = self._normalize_scores(numpy.asarray(raw_output).squeeze())
        top_indices = numpy.argsort(scores)[-3:][::-1]
        predictions = [
            TagPrediction(
                label=self.labels[class_index],
                score=round(float(scores[class_index]), 4),
            )
            for class_index in top_indices
        ]

        result = InferenceResult(
            model_name=self.model_name,
            tags=predictions,
            width=width,
            height=height,
            preprocessing_ms=preprocessing_ms,
            model_ms=model_ms,
            cache_hit=False,
        )

        if _cache is not None:
            _cache.set(cache_key, _serialize_result(result), ex=_INFERENCE_CACHE_TTL)

        return result

    def _build_input_tensor(self, image: Image.Image) -> numpy.ndarray:
        fitted_image = ImageOps.fit(
            image, 
            size=IMAGE_INPUT_SIZE, 
            method=Image.Resampling.BILINEAR
        )
        image_array = numpy.asarray(fitted_image, dtype=numpy.float32) / 255.0
        normalized_array = (image_array - IMAGE_MEAN) / IMAGE_STD
        chw_array = numpy.transpose(normalized_array, (2, 0, 1))
        return numpy.expand_dims(chw_array, axis=0).astype(numpy.float32)

    def _normalize_scores(self, model_output: numpy.ndarray) -> numpy.ndarray:
        if numpy.all(model_output >= 0) and numpy.isclose(float(model_output.sum()), 1.0, atol=1e-3):
            return model_output

        shifted_output = model_output - numpy.max(model_output)
        exponentiated_output = numpy.exp(shifted_output)
        return exponentiated_output / exponentiated_output.sum()


@lru_cache(maxsize=1)
def get_pretrained_image_classifier() -> PretrainedImageClassifier:
    return PretrainedImageClassifier()
