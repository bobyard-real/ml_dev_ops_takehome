import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ml_dev_ops_takehome.settings")

import django

django.setup()

from django.test import RequestFactory

from inference.views import home


def test_home_get_returns_200():
    factory = RequestFactory()
    request = factory.get("/")
    response = home(request)
    assert response.status_code == 200


def test_health_returns_ok():
    from django.test import Client

    client = Client()
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint():
    from django.test import Client

    client = Client()
    response = client.get("/metrics/")
    assert response.status_code == 200
    assert b"inference_model_loaded" in response.content
