import io
from unittest.mock import MagicMock

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image

from inference.services import InferenceResult, TagPrediction


MOCK_RESULT = InferenceResult(
    model_name="squeezenet1_1.onnx",
    tags=[
        TagPrediction(label="tabby cat", score=0.8231),
        TagPrediction(label="tiger cat", score=0.1102),
        TagPrediction(label="Egyptian cat", score=0.0421),
    ],
    width=640,
    height=480,
    preprocessing_ms=2.1,
    model_ms=18.4,
    cache_hit=False,
)


@pytest.fixture
def mock_classifier(monkeypatch):
    """Replaces the real classifier and telemetry with mocks."""
    mock = MagicMock()
    mock.classify.return_value = MOCK_RESULT
    monkeypatch.setattr("inference.views.get_pretrained_image_classifier", lambda: mock)
    monkeypatch.setattr("inference.views.record_inference_metrics", MagicMock())
    return mock


@pytest.fixture
def png_image():
    """Minimal valid PNG as a named upload file Django's ImageField accepts."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(100, 149, 237)).save(buf, format="PNG")
    return SimpleUploadedFile("test.png", buf.getvalue(), content_type="image/png")
