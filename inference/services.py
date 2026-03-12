import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import BinaryIO

import numpy
import onnxruntime
import xxhash
from PIL import Image, ImageOps

from inference.assets import LABELS_PATH, MODEL_PATH, ensure_model_assets
from inference.metrics import (
    CACHE_HITS,
    CACHE_MISSES,
    CACHE_SIZE,
    INFERENCE_LATENCY,
    MODEL_LOADED,
)


logger = logging.getLogger(__name__)

IMAGE_INPUT_SIZE = (224, 224)
IMAGE_MEAN = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
IMAGE_STD = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)

# Cache config — tunable via environment variables
CACHE_MAX_SIZE = int(os.environ.get("INFERENCE_CACHE_MAX_SIZE", "256"))

# Thread config — defaults optimized for typical CPU deployment
INTRA_OP_THREADS = int(os.environ.get("ORT_INTRA_OP_THREADS", "0"))  # 0 = ORT auto-detects
INTER_OP_THREADS = int(os.environ.get("ORT_INTER_OP_THREADS", "1"))


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


class PretrainedImageClassifier:
    def __init__(self) -> None:
        ensure_model_assets()
        self.model_name = MODEL_PATH.name
        self.labels = tuple(
            label.strip()
            for label in LABELS_PATH.read_text().splitlines()
            if label.strip()
        )

        # --- Session tuning ---
        sess_options = onnxruntime.SessionOptions()
        sess_options.intra_op_num_threads = INTRA_OP_THREADS
        sess_options.inter_op_num_threads = INTER_OP_THREADS
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.enable_mem_pattern = True
        # Cache the optimized graph to disk so ORT skips re-optimization on next load
        optimized_path = MODEL_PATH.with_suffix(".optimized.onnx")
        sess_options.optimized_model_filepath = optimized_path.as_posix()
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = onnxruntime.InferenceSession(
            MODEL_PATH.as_posix(),
            sess_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

        # --- Warm-start: trigger lazy allocations before first real request ---
        dummy_input = numpy.zeros((1, 3, *IMAGE_INPUT_SIZE), dtype=numpy.float32)
        self.session.run(None, {self.input_name: dummy_input})
        MODEL_LOADED.set(1)
        logger.info(
            "Model warm-start complete: %s (threads: intra=%d, inter=%d)",
            self.model_name,
            INTRA_OP_THREADS,
            INTER_OP_THREADS,
        )

        # Pre-compute fused normalization constants to avoid per-request division
        self._scale = (1.0 / (255.0 * IMAGE_STD)).astype(numpy.float32)
        self._shift = (IMAGE_MEAN / IMAGE_STD).astype(numpy.float32)

        # --- Result cache: input hash → InferenceResult ---
        self._cache: OrderedDict[str, InferenceResult] = OrderedDict()

    def classify(self, uploaded_image: BinaryIO) -> InferenceResult:
        uploaded_image.seek(0)
        image_bytes = uploaded_image.read()

        # Check cache before doing any work
        cache_key = xxhash.xxh64(image_bytes).hexdigest()
        if cache_key in self._cache:
            CACHE_HITS.inc()
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        CACHE_MISSES.inc()

        with Image.open(BytesIO(image_bytes)) as image:
            normalized_image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = normalized_image.size
            input_tensor = self._build_input_tensor(normalized_image)

        inf_start = time.monotonic()
        raw_output = self.session.run(None, {self.input_name: input_tensor})[0]
        INFERENCE_LATENCY.observe(time.monotonic() - inf_start)
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
        )

        # Store in cache, evict oldest if full
        self._cache[cache_key] = result
        if len(self._cache) > CACHE_MAX_SIZE:
            self._cache.popitem(last=False)
        CACHE_SIZE.set(len(self._cache))

        return result

    def _build_input_tensor(self, image: Image.Image) -> numpy.ndarray:
        fitted_image = ImageOps.fit(
            image,
            size=IMAGE_INPUT_SIZE,
            method=Image.Resampling.BILINEAR,
        )
        # Single-pass normalize: divide, subtract mean, divide std, transpose, add batch dim
        # Using pre-computed 1/(255*std) and mean/std avoids two separate array operations
        image_array = numpy.asarray(fitted_image, dtype=numpy.float32)
        image_array *= self._scale  # 1.0 / (255.0 * std) — fused scale+normalize
        image_array -= self._shift  # mean / std
        return numpy.ascontiguousarray(
            image_array.transpose(2, 0, 1)[numpy.newaxis]
        )

    def _normalize_scores(self, model_output: numpy.ndarray) -> numpy.ndarray:
        if numpy.all(model_output >= 0) and numpy.isclose(float(model_output.sum()), 1.0, atol=1e-3):
            return model_output

        shifted_output = model_output - numpy.max(model_output)
        exponentiated_output = numpy.exp(shifted_output)
        return exponentiated_output / exponentiated_output.sum()


@lru_cache(maxsize=1)
def get_pretrained_image_classifier() -> PretrainedImageClassifier:
    return PretrainedImageClassifier()
