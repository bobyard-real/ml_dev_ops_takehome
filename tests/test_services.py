import json

import pytest

from inference.services import (
    InferenceResult,
    TagPrediction,
    _deserialize_result,
    _serialize_result,
)
from tests.conftest import MOCK_RESULT


class TestInferenceResultSerialization:
    def test_round_trip(self):
        data = _serialize_result(MOCK_RESULT)
        restored = _deserialize_result(data)

        assert restored.model_name == MOCK_RESULT.model_name
        assert restored.width == MOCK_RESULT.width
        assert restored.height == MOCK_RESULT.height
        assert restored.preprocessing_ms == MOCK_RESULT.preprocessing_ms
        assert restored.model_ms == MOCK_RESULT.model_ms
        assert len(restored.tags) == len(MOCK_RESULT.tags)
        for original, restored_tag in zip(MOCK_RESULT.tags, restored.tags):
            assert restored_tag.label == original.label
            assert restored_tag.score == original.score

    def test_cache_hit_false_on_serialize(self):
        """cache_hit must never be stored as True — avoids stale flag in cache."""
        cached = InferenceResult(
            model_name="model.onnx",
            tags=[TagPrediction(label="cat", score=1.0)],
            width=10, height=10,
            preprocessing_ms=0.0, model_ms=0.0,
            cache_hit=True,
        )
        data = json.loads(_serialize_result(cached))
        assert data["cache_hit"] is False

    def test_cache_hit_true_on_deserialize(self):
        """Deserialized results must always be marked as cache hits."""
        restored = _deserialize_result(_serialize_result(MOCK_RESULT))
        assert restored.cache_hit is True

    def test_tags_preserved_in_order(self):
        data = _serialize_result(MOCK_RESULT)
        restored = _deserialize_result(data)
        labels = [t.label for t in restored.tags]
        assert labels == ["tabby cat", "tiger cat", "Egyptian cat"]
