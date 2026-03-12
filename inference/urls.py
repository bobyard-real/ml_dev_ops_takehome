from django.urls import path

from inference import views


app_name = "inference"


urlpatterns = [
    path("", views.home, name="home"),
    path("api/infer/", views.infer_image, name="infer_image"),
    path("health/", views.health, name="health"),
    path("ready/", views.ready, name="ready"),
    path("metrics/", views.metrics, name="metrics"),
]
