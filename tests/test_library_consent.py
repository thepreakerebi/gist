import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gist.access.store import Principal
from gist.api.app import create_app
from gist.api.auth import current_principal
from gist.api.library import AddVideoRequest

URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"


def test_add_video_request_requires_consent() -> None:
    with pytest.raises(ValidationError, match="privacy notice"):
        AddVideoRequest(url=URL)

    with pytest.raises(ValidationError, match="privacy notice"):
        AddVideoRequest(url=URL, consent=False)

    assert AddVideoRequest(url=URL, consent=True).consent is True


def test_add_video_endpoint_rejects_missing_consent_before_touching_the_database() -> None:
    # Validation runs before the handler, so this holds with or without
    # DATABASE_URL: no row is created and no ingestion starts.
    app = create_app()
    app.dependency_overrides[current_principal] = lambda: Principal(
        code_id="00000000-0000-0000-0000-000000000001", label="test", is_admin=False
    )
    client = TestClient(app)

    response = client.post("/v1/library/videos", json={"url": URL})

    assert response.status_code == 422
    assert "privacy notice" in response.text
