from fastapi.testclient import TestClient

from gist.api.app import create_app


def test_health_endpoint() -> None:
    client = TestClient(create_app())

    response = client.get("/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_compression_endpoint() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/compressions",
        json={
            "video_id": "demo",
            "query": "pricing",
            "duration_seconds": 90,
            "preset": "balanced",
            "visual_candidates": [
                {"id": "v1", "timestamp_seconds": 11, "text": "pricing slide"},
            ],
            "audio_candidates": [
                {"id": "a1", "timestamp_seconds": 12, "text": "pricing is explained"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == "demo"
    assert body["metrics"]["selected_candidates"] == 2

