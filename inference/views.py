import logging
import time

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from prometheus_client import generate_latest

from inference.forms import ImageUploadForm
from inference.metrics import MODEL_LOADED, REQUEST_COUNT, REQUEST_LATENCY
from inference.services import InferenceResult, get_pretrained_image_classifier


def _serialize_result(result: InferenceResult) -> dict[str, object]:
    return {
        "model": result.model_name,
        "image": {
            "width": result.width,
            "height": result.height,
        },
        "tags": [
            {
                "label": prediction.label,
                "score": prediction.score,
            }
            for prediction in result.tags
        ],
    }


@require_http_methods(["GET", "POST"])
def home(request: HttpRequest) -> HttpResponse:
    form = ImageUploadForm(request.POST or None, request.FILES or None)
    result_payload = None

    if request.method == "POST" and form.is_valid():
        start = time.monotonic()
        result = get_pretrained_image_classifier().classify(form.cleaned_data["image"])
        REQUEST_LATENCY.labels(endpoint="home").observe(time.monotonic() - start)
        REQUEST_COUNT.labels(endpoint="home", status="200").inc()
        result_payload = _serialize_result(result)
        logging.info(
            "Completed browser inference request model=%s tags=%s",
            result.model_name,
            ",".join(tag["label"] for tag in result_payload["tags"]),
        )

    return render(
        request,
        "inference/home.html",
        {
            "form": form,
            "result": result_payload,
        },
    )


@csrf_exempt
@require_POST
def infer_image(request: HttpRequest) -> JsonResponse:
    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        REQUEST_COUNT.labels(endpoint="api_infer", status="400").inc()
        return JsonResponse({"errors": form.errors.get_json_data()}, status=400)

    start = time.monotonic()
    image = form.cleaned_data["image"]
    result = get_pretrained_image_classifier().classify(image)
    REQUEST_LATENCY.labels(endpoint="api_infer").observe(time.monotonic() - start)
    REQUEST_COUNT.labels(endpoint="api_infer", status="200").inc()
    result_payload = _serialize_result(result)

    logging.info(
        "Completed API inference request model=%s tags=%s",
        result.model_name,
        ",".join(tag["label"] for tag in result_payload["tags"]),
    )

    return JsonResponse(result_payload)


@require_GET
def health(request: HttpRequest) -> JsonResponse:
    """Liveness check — is the process up and responding."""
    return JsonResponse({"status": "ok"})


@require_GET
def ready(request: HttpRequest) -> JsonResponse:
    """Readiness check — is the model loaded and warm."""
    if MODEL_LOADED._value.get() == 0:
        return JsonResponse({"status": "not ready"}, status=503)
    return JsonResponse({"status": "ready", "model": "squeezenet1.1-7.onnx"})


@require_GET
def metrics(request: HttpRequest) -> HttpResponse:
    """Prometheus-compatible metrics endpoint."""
    return HttpResponse(generate_latest(), content_type="text/plain; charset=utf-8")
