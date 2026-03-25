import pytest
from django.test import Client

from tests.conftest import MOCK_RESULT


@pytest.fixture
def client():
    return Client()


@pytest.mark.django_db
class TestInferImageEndpoint:
    URL = "/api/infer/"

    def test_valid_image_returns_200(self, client, mock_classifier, png_image):
        response = client.post(self.URL, {"image": png_image}, format="multipart")
        assert response.status_code == 200

    def test_response_schema(self, client, mock_classifier, png_image):
        response = client.post(self.URL, {"image": png_image}, format="multipart")
        data = response.json()

        assert "model" in data
        assert "image" in data
        assert "width" in data["image"]
        assert "height" in data["image"]
        assert "tags" in data
        assert len(data["tags"]) == 3
        for tag in data["tags"]:
            assert "label" in tag
            assert "score" in tag

    def test_response_values_match_classifier_output(self, client, mock_classifier, png_image):
        response = client.post(self.URL, {"image": png_image}, format="multipart")
        data = response.json()

        assert data["model"] == MOCK_RESULT.model_name
        assert data["image"]["width"] == MOCK_RESULT.width
        assert data["image"]["height"] == MOCK_RESULT.height
        assert data["tags"][0]["label"] == MOCK_RESULT.tags[0].label
        assert data["tags"][0]["score"] == MOCK_RESULT.tags[0].score

    def test_missing_image_returns_400(self, client, mock_classifier):
        response = client.post(self.URL, {}, format="multipart")
        assert response.status_code == 400
        assert "errors" in response.json()

    def test_get_method_not_allowed(self, client):
        response = client.get(self.URL)
        assert response.status_code == 405
