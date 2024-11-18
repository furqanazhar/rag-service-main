from fastapi.testclient import TestClient
from fastapi_service.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
