"""
Observability helpers: structured logging + CloudWatch EMF metrics.

Metrics are emitted via Embedded Metric Format (EMF) — annotated JSON written
to stdout. ECS ships stdout to CloudWatch Logs, which automatically extracts the
metrics into CloudWatch Metrics under the namespace MLApp/Inference.
No extra infrastructure required.
"""

import logging
import os

from aws_embedded_metrics import metric_scope
from aws_embedded_metrics.unit import Unit

logger = logging.getLogger(__name__)

_NAMESPACE = "MLApp/Inference"
_ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")


@metric_scope
async def _emit(
    total_ms: float,
    preprocessing_ms: float,
    model_ms: float,
    image_pixels: int,
    top_score: float,
    metrics,  # injected by @metric_scope
) -> None:
    metrics.set_namespace(_NAMESPACE)
    metrics.put_dimensions({"Environment": _ENVIRONMENT})
    metrics.put_metric("InferenceDurationMs", total_ms, Unit.MILLISECONDS)
    metrics.put_metric("PreprocessingDurationMs", preprocessing_ms, Unit.MILLISECONDS)
    metrics.put_metric("ModelDurationMs", model_ms, Unit.MILLISECONDS)
    metrics.put_metric("ImagePixels", image_pixels, Unit.COUNT)
    metrics.put_metric("TopScore", top_score, Unit.NONE)


def record_inference_metrics(
    *,
    total_ms: float,
    preprocessing_ms: float,
    model_ms: float,
    image_width: int,
    image_height: int,
    top_score: float,
    top_label: str,
    model_name: str,
    source: str,
) -> None:
    """Emit EMF metrics and a structured log line for a completed inference request."""
    import asyncio

    asyncio.run(
        _emit(
            total_ms=total_ms,
            preprocessing_ms=preprocessing_ms,
            model_ms=model_ms,
            image_pixels=image_width * image_height,
            top_score=top_score,
        )
    )

    logger.info(
        "inference_complete",
        extra={
            "source": source,
            "model_name": model_name,
            "total_ms": round(total_ms, 2),
            "preprocessing_ms": round(preprocessing_ms, 2),
            "model_ms": round(model_ms, 2),
            "image_width": image_width,
            "image_height": image_height,
            "top_label": top_label,
            "top_score": top_score,
        },
    )
