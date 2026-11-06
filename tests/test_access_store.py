"""The access SQL against a real Postgres.

Skipped unless GIST_TEST_DATABASE_URL points at a *throwaway* database with
pgvector, for example the end-to-end container:

    docker run -d --name gist-e2e-pg -e POSTGRES_PASSWORD=gist -e POSTGRES_DB=gist \\
      -p 55432:5432 pgvector/pgvector:pg16
    GIST_TEST_DATABASE_URL=postgresql://postgres:gist@127.0.0.1:55432/gist \\
      uv run pytest tests/test_access_store.py

Each test drops and recreates the schema, so never point it at real data.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta

import pytest

from gist.access import store
from gist.db import connection as db
from gist.db import repository as repo

URL = os.getenv("GIST_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="GIST_TEST_DATABASE_URL is not set")


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DATABASE_URL", URL or "")
    db.close_pool()
    with db.cursor() as cur:
        cur.execute(
            "drop table if exists messages, conversations, audio_windows, frames, videos, "
            "question_usage, sessions, access_codes cascade"
        )
    db.apply_schema()
    yield
    db.close_pool()


def _add(url: str, owner: str) -> repo.VideoRecord:
    return repo.create_video(url, title="t", consent_version="v", owner_code_id=owner)


def test_a_code_signs_in_and_out_everywhere_and_back_in() -> None:
    code, code_id = store.create_code("Marvin")
    first = store.login(code.lower().replace("-", " "))
    second = store.login(code)
    assert first and second
    assert store.principal_for_token(first[0]) == store.Principal(code_id, "Marvin", False)

    assert store.end_all_sessions(code_id) == 2
    assert store.principal_for_token(first[0]) is None
    assert store.principal_for_token(second[0]) is None

    again = store.login(code)
    assert again and store.principal_for_token(again[0]) is not None


def test_wrong_revoked_and_expired_codes_do_not_sign_in() -> None:
    assert store.login("GIST-0000-0000-0000-0000") is None

    code, code_id = store.create_code("x")
    token, _ = store.login(code) or ("", None)
    store.revoke_code(code_id)
    assert store.login(code) is None
    assert store.principal_for_token(token) is None

    old, _ = store.create_code("y", expires_at=datetime.now(UTC) - timedelta(seconds=1))
    assert store.login(old) is None


def test_the_code_itself_is_never_stored() -> None:
    code, _ = store.create_code("x")
    with db.cursor() as cur:
        cur.execute("select * from access_codes")
        row = cur.fetchone()
    assert code not in str(row) and store.normalize_code(code) not in str(row)


def test_the_daily_cap_stops_at_five_and_refunds() -> None:
    _, code_id = store.create_code("x")
    day = date(2026, 11, 6)
    assert [store.charge_question(code_id, day=day, cap=5) for _ in range(6)] == [True] * 5 + [
        False
    ]
    assert store.questions_used(code_id, day=day) == 5

    store.refund_question(code_id, day=day)
    assert store.charge_question(code_id, day=day, cap=5) is True
    assert store.charge_question(code_id, day=date(2026, 11, 7), cap=5) is True


def test_videos_are_per_owner_and_transfer_to_a_replacement_code() -> None:
    _, alice = store.create_code("alice")
    _, bob = store.create_code("bob")
    url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"

    a = _add(url, alice)
    assert _add(url, alice).id == a.id  # the same owner re-adding is an upsert
    b = _add(url, bob)
    assert b.id != a.id  # another owner gets a separate copy

    assert [v.id for v in repo.list_videos(owner_code_id=alice)] == [a.id]
    assert len(repo.list_videos(owner_code_id=None)) == 2
    assert repo.get_video_by_url(url, owner_code_id=bob).id == b.id

    _, replacement = store.create_code("alice, new code")
    assert store.transfer_videos(alice, replacement) == 1
    assert repo.get_video(a.id).owner_code_id == replacement


def test_replacing_a_lost_code_keeps_the_videos_and_revokes_the_old_one() -> None:
    expires = datetime.now(UTC) + timedelta(days=30)
    old_code, old_id = store.create_code("Marvin", expires_at=expires)
    token, _ = store.login(old_code) or ("", None)
    video = _add("https://lost", old_id)

    new_code, new_id, moved = store.replace_code(old_id)

    assert moved == 1 and new_code != old_code
    assert repo.get_video(video.id).owner_code_id == new_id
    assert store.login(old_code) is None
    assert store.principal_for_token(token) is None
    signed_in = store.login(new_code)
    assert signed_in and signed_in[1] == store.Principal(new_id, "Marvin", False)
    replaced = next(row for row in store.list_codes() if row.id == new_id)
    assert replaced.expires_at == expires  # replacing does not extend a code's life


def test_replacing_an_unknown_code_changes_nothing() -> None:
    with pytest.raises(LookupError):
        store.replace_code("00000000-0000-0000-0000-000000000000")
    assert store.list_codes() == []


def test_an_admin_claims_videos_from_before_codes_existed() -> None:
    with db.cursor() as cur:
        cur.execute(
            "insert into videos (youtube_url, title, duration_seconds) values ('u', 't', 0)"
        )
    _, admin = store.create_code("me", is_admin=True)
    assert store.claim_unowned_videos(admin) == 1
    assert repo.list_videos(owner_code_id=admin)[0].owner_code_id == admin


def test_the_schema_migrates_a_database_from_before_access_codes() -> None:
    # The old shape: youtube_url globally unique and no owner column.
    with db.cursor() as cur:
        cur.execute(
            "drop table if exists messages, conversations, audio_windows, frames, videos cascade"
        )
        cur.execute(
            "create table videos (id uuid primary key default gen_random_uuid(), "
            "youtube_url text not null unique, youtube_id text, title text not null, "
            "duration_seconds double precision not null, thumbnail_url text, source_path text, "
            "status text not null default 'pending', status_detail text, "
            "progress double precision not null default 0, frame_count integer not null default 0, "
            "audio_window_count integer not null default 0, transcript text, error text, "
            "created_at timestamptz not null default now(), ingested_at timestamptz)"
        )
    db.apply_schema()
    db.apply_schema()  # idempotent

    _, alice = store.create_code("alice")
    _, bob = store.create_code("bob")
    _add("https://same", alice)
    _add("https://same", bob)
    assert len(repo.list_videos(owner_code_id=None)) == 2
