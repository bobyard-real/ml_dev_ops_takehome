from django.apps import AppConfig


class InferenceConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "inference"

    def ready(self) -> None:
        """Eagerly load the model at startup so the first request isn't slow."""
        from inference.services import get_pretrained_image_classifier

        get_pretrained_image_classifier()
