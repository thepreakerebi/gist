"""Access codes, rate limits and the access rules on the HTTP surface.

These run without a database: the store is replaced where a route reaches it.
``test_access_store.py`` exercises the SQL itself against a real Postgres.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from gist.access import codes, store
from gist.access.ratelimit import Limit, RateLimiter, limiter
from gist.access.store import Principal
from gist.api import app as app_module
from gist.api import auth, library
from gist.api.app import create_app
from gist.db import repository as repo

ORIGIN = "http://localhost:3000"
OWNER = Principal(code_id="11111111-1111-1111-1111-111111111111", label="owner", is_admin=False)
OTHER = Principal(code_id="22222222-2222-2222-2222-222222222222", label="other", is_admin=False)
ADMIN = Principal(code_id="33333333-3333-3333-3333-333333333333", label="admin", is_admin=True)
VIDEO_ID = "44444444-4444-4444-4444-444444444444"


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter.reset()
    monkeypatch.delenv("GIST_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("GIST_ENABLE_RESEARCH_ROUTES", raising=False)


def _video(owner: str | None, status: str = "ready") -> repo.VideoRecord:
    return repo.VideoRecord(
        id=VIDEO_ID,
        youtube_url="https://www.youtube.com/watch?v=x",
        youtube_id="x",
        title="t",
        duration_seconds=60.0,
        thumbnail_url=None,
        source_path=None,
        status=status,
        status_detail=None,
        progress=1.0,
        frame_count=1,
        audio_window_count=1,
        error=None,
        created_at=datetime.now(UTC),
        ingested_at=None,
        owner_code_id=owner,
    )


def _client_as(principal: Principal, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(library.db, "is_configured", lambda: True)
    app = create_app()
    app.dependency_overrides[auth.current_principal] = lambda: principal
    return TestClient(app)


# ------------------------------------------------------------------ codes ----


def test_generated_codes_have_the_display_shape_and_round_trip() -> None:
    code = codes.generate_code()
    prefix, *groups = code.split("-")
    assert prefix == "GIST"
    assert [len(group) for group in groups] == [4, 4, 4, 4]
    assert all(ch in codes.ALPHABET for ch in "".join(groups))
    assert codes.normalize_code(code) == "".join(groups)
    assert codes.generate_code() != code


def test_typed_codes_are_normalised_forgivingly() -> None:
    expected = "7K3M0X9PR2TAH8WD"
    for typed in (
        "GIST-7K3M-0X9P-R2TA-H8WD",
        "gist 7k3m 0x9p r2ta h8wd",
        "7K3M-OX9P-R2TA-H8WD",  # letter O for zero
        "  7k3m0x9pr2tah8wd  ",
    ):
        assert codes.normalize_code(typed) == expected


@pytest.mark.parametrize(
    "typed", ["", "GIST", "7K3M-0X9P", "7K3M0X9PR2TAH8WDX", "7K3M0X9PR2TAH8WU"]
)
def test_malformed_codes_are_rejected_before_any_lookup(typed: str) -> None:
    assert codes.normalize_code(typed) is None


def test_only_hashes_are_comparable_and_tokens_are_long() -> None:
    assert codes.hash_secret("A") != codes.hash_secret("B")
    assert len(codes.hash_secret("A")) == 64
    assert len(codes.generate_session_token()) >= 43


def test_cap_resets_at_midnight_kigali() -> None:
    late = datetime(2026, 11, 6, 21, 59, tzinfo=UTC)  # 23:59 in Kigali
    after = datetime(2026, 11, 6, 22, 1, tzinfo=UTC)  # 00:01 the next day
    assert store.cap_day(late).isoformat() == "2026-11-06"
    assert store.cap_day(after).isoformat() == "2026-11-07"
    assert store.cap_resets_at(late).isoformat() == "2026-11-07T00:00:00+02:00"


# -------------------------------------------------------------- limiter ----


def test_rate_limiter_refuses_past_the_limit_and_frees_up_with_time() -> None:
    now = [0.0]
    limits = RateLimiter(clock=lambda: now[0])
    rule = Limit(3, 60)

    assert [limits.hit("k", rule) for _ in range(3)] == [None, None, None]
    assert limits.hit("k", rule) == 60
    assert limits.hit("other", rule) is None

    now[0] = 59.0
    assert limits.hit("k", rule) == 1  # refused attempts were not recorded
    now[0] = 60.5
    assert limits.hit("k", rule) is None


def test_rate_limiter_applies_every_window() -> None:
    now = [0.0]
    limits = RateLimiter(clock=lambda: now[0])
    rules = (Limit(2, 10), Limit(3, 100))
    for step in range(3):
        now[0] = step * 11.0
        assert limits.hit("k", *rules) is None
    now[0] = 40.0
    assert limits.hit("k", *rules) == 60  # the hourly-style window, not the short one


# ------------------------------------------------------------- the app ----


def test_library_requires_a_session() -> None:
    client = TestClient(create_app())
    for method, path in (
        ("get", "/v1/library/videos"),
        ("get", f"/v1/library/videos/{VIDEO_ID}"),
        ("get", f"/v1/library/videos/{VIDEO_ID}/source"),
        ("get", f"/v1/library/videos/{VIDEO_ID}/clips/a.mp4"),
        ("delete", f"/v1/library/videos/{VIDEO_ID}"),
    ):
        response = getattr(client, method)(path)
        assert response.status_code == 401, path
    assert client.get("/v1/auth/me").status_code == 401


def test_health_stays_public() -> None:
    assert TestClient(create_app()).get("/v1/health").status_code == 200


def test_research_routes_are_not_mounted_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    client = TestClient(create_app())
    for path in (
        "/v1/ingestions",
        "/v1/local-video-compressions",
        "/v1/compressions",
        "/v1/demo/run",
    ):
        assert client.post(path, json={}).status_code == 404, path

    monkeypatch.setenv("GIST_ENABLE_RESEARCH_ROUTES", "1")
    assert TestClient(create_app()).post("/v1/ingestions", json={}).status_code == 422


def test_wildcard_cors_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIST_CORS_ORIGINS", "*")
    with pytest.raises(ValueError, match="cannot be"):
        app_module.allowed_origins()


def test_cross_site_writes_are_refused_before_authentication() -> None:
    client = TestClient(create_app())
    response = client.post(
        "/v1/auth/login", json={"code": "x"}, headers={"Origin": "https://evil.example"}
    )
    assert response.status_code == 403


def test_cors_allows_the_frontend_with_credentials() -> None:
    response = TestClient(create_app()).options(
        "/v1/library/videos",
        headers={"Origin": ORIGIN, "Access-Control-Request-Method": "GET"},
    )
    assert response.headers["access-control-allow-origin"] == ORIGIN
    assert response.headers["access-control-allow-credentials"] == "true"


def test_a_video_id_must_be_a_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client_as(OWNER, monkeypatch)
    for bad in ("not-a-uuid", "%2e%2e", "...."):
        assert client.get(f"/v1/library/videos/{bad}/frames/a.jpg").status_code == 422, bad
    # A bare ".." is resolved by the client before sending; refused either way.
    assert client.get("/v1/library/videos/../frames/a.jpg").status_code in {404, 422}


def test_another_codes_video_looks_like_it_does_not_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(repo, "get_video", lambda video_id: _video(OWNER.code_id))
    monkeypatch.setattr(repo, "latest_conversation", lambda video_id: None)

    assert _client_as(OTHER, monkeypatch).get(f"/v1/library/videos/{VIDEO_ID}").status_code == 404
    assert (
        _client_as(OTHER, monkeypatch).delete(f"/v1/library/videos/{VIDEO_ID}").status_code == 404
    )
    assert _client_as(OWNER, monkeypatch).get(f"/v1/library/videos/{VIDEO_ID}").status_code == 200
    assert _client_as(ADMIN, monkeypatch).get(f"/v1/library/videos/{VIDEO_ID}").status_code == 200


def test_the_library_lists_only_the_callers_videos(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def fake_list(*, owner_code_id: str | None, limit: int = 100) -> list[repo.VideoRecord]:
        seen.append(owner_code_id)
        return []

    monkeypatch.setattr(repo, "list_videos", fake_list)
    _client_as(OWNER, monkeypatch).get("/v1/library/videos")
    _client_as(ADMIN, monkeypatch).get("/v1/library/videos")
    assert seen == [OWNER.code_id, None]


def test_the_daily_cap_refuses_a_question(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repo, "get_video", lambda video_id: _video(OWNER.code_id))
    monkeypatch.setattr(store, "charge_question", lambda code_id, *, day, cap: False)

    response = _client_as(OWNER, monkeypatch).post(
        f"/v1/library/videos/{VIDEO_ID}/query", json={"query": "what happens"}
    )
    assert response.status_code == 429
    assert "all 5 questions for today" in response.json()["detail"]


def test_admins_have_no_daily_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repo, "get_video", lambda video_id: _video(ADMIN.code_id))

    def never(*args: Any, **kwargs: Any) -> bool:
        raise AssertionError("an admin question must not be charged")

    monkeypatch.setattr(store, "charge_question", never)
    monkeypatch.setattr(library, "_query_stream", lambda record, request, refund: iter(()))
    response = _client_as(ADMIN, monkeypatch).post(
        f"/v1/library/videos/{VIDEO_ID}/query", json={"query": "what happens"}
    )
    assert response.status_code == 200


def test_a_failed_question_is_given_back(monkeypatch: pytest.MonkeyPatch) -> None:
    refunded: list[bool] = []
    events: list[tuple[str, dict[str, Any]]] = []

    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("scoring failed")

    monkeypatch.setattr(repo, "ensure_conversation", broken)
    library._query_worker(
        _video(OWNER.code_id),
        library.QueryRequest(query="q"),
        lambda event, data: events.append((event, data)),
        lambda: refunded.append(True),
    )
    assert refunded == [True]
    assert events[0] == ("error", {"message": "scoring failed"})


def test_login_sets_a_locked_down_cookie(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth.db, "is_configured", lambda: True)
    monkeypatch.setattr(store, "login", lambda code: ("token-value", OWNER))
    monkeypatch.setattr(store, "questions_used", lambda code_id, *, day: 2)

    response = TestClient(create_app()).post("/v1/auth/login", json={"code": "GIST-..."})
    assert response.status_code == 200
    assert response.json()["questions_left_today"] == 3
    cookie = response.headers["set-cookie"].lower()
    assert cookie.startswith(f"{auth.COOKIE_NAME}=token-value")
    assert "httponly" in cookie and "samesite=lax" in cookie and "max-age=86400" in cookie


def test_login_attempts_are_rate_limited(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth.db, "is_configured", lambda: True)
    monkeypatch.setattr(store, "login", lambda code: None)
    client = TestClient(create_app())

    statuses = [client.post("/v1/auth/login", json={"code": "wrong"}).status_code for _ in range(6)]
    assert statuses == [401] * 5 + [429]


def test_a_session_cookie_identifies_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(auth.db, "is_configured", lambda: True)
    monkeypatch.setattr(
        store, "principal_for_token", lambda token: OWNER if token == "good" else None
    )
    monkeypatch.setattr(store, "questions_used", lambda code_id, *, day: 0)
    client = TestClient(create_app())

    client.cookies.set(auth.COOKIE_NAME, "good")
    assert client.get("/v1/auth/me").json()["label"] == "owner"
    client.cookies.set(auth.COOKIE_NAME, "stale")
    assert client.get("/v1/auth/me").status_code == 401


def test_logout_ends_every_session_of_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    ended: list[str] = []
    monkeypatch.setattr(auth.db, "is_configured", lambda: True)
    monkeypatch.setattr(store, "principal_for_token", lambda token: OWNER)
    monkeypatch.setattr(store, "end_all_sessions", lambda code_id: ended.append(code_id) or 2)
    client = TestClient(create_app())
    client.cookies.set(auth.COOKIE_NAME, "good")

    response = client.post("/v1/auth/logout")
    assert response.status_code == 204
    assert ended == [OWNER.code_id]
    assert "max-age=0" in response.headers["set-cookie"].lower()
