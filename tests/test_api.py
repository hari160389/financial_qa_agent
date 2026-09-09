from fastapi.testclient import TestClient

from src.api.app import app


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_records():
    client = TestClient(app)
    response = client.get("/records", params={"split": "dev", "limit": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
