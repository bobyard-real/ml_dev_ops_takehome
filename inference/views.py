import logging
import time

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods, require_POST

from inference.forms import ImageUploadForm
from inference.services import InferenceResult, get_pretrained_image_classifier
from inference.telemetry import record_inference_metrics

logger = logging.getLogger(__name__)


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
        t0 = time.perf_counter()
        result = get_pretrained_image_classifier().classify(form.cleaned_data["image"])
        total_ms = (time.perf_counter() - t0) * 1000

        result_payload = _serialize_result(result)
        record_inference_metrics(
            total_ms=total_ms,
            preprocessing_ms=result.preprocessing_ms,
            model_ms=result.model_ms,
            image_width=result.width,
            image_height=result.height,
            top_label=result.tags[0].label,
            top_score=result.tags[0].score,
            model_name=result.model_name,
            source="browser",
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
        return JsonResponse({"errors": form.errors.get_json_data()}, status=400)

    t0 = time.perf_counter()
    result = get_pretrained_image_classifier().classify(form.cleaned_data["image"])
    total_ms = (time.perf_counter() - t0) * 1000

    result_payload = _serialize_result(result)
    record_inference_metrics(
        total_ms=total_ms,
        preprocessing_ms=result.preprocessing_ms,
        model_ms=result.model_ms,
        image_width=result.width,
        image_height=result.height,
        top_label=result.tags[0].label,
        top_score=result.tags[0].score,
        model_name=result.model_name,
        source="api",
    )

    return JsonResponse(result_payload)
