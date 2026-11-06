"""Database access for codes, sessions and the daily question cap."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from gist.access.codes import generate_code, generate_session_token, hash_secret, normalize_code
from gist.db.connection import cursor

SESSION_LIFETIME = timedelta(days=1)
# Questions a non-admin code may ask per day.
DAILY_QUESTION_CAP = int(os.getenv("GIST_DAILY_QUESTION_CAP", "5"))
# The daily cap resets at midnight in this zone.
CAP_TIMEZONE = ZoneInfo(os.getenv("GIST_CAP_TIMEZONE", "Africa/Kigali"))


@dataclass(frozen=True, slots=True)
class Principal:
    """The access code a request is acting as."""

    code_id: str
    label: str
    is_admin: bool


@dataclass(frozen=True, slots=True)
class CodeRecord:
    id: str
    label: str
    is_admin: bool
    created_at: datetime
    expires_at: datetime | None
    revoked_at: datetime | None
    video_count: int
    active_sessions: int


# ------------------------------------------------------------------ codes ----


def create_code(
    label: str, *, is_admin: bool = False, expires_at: datetime | None = None
) -> tuple[str, str]:
    """Create a code and return ``(display_code, code_id)``.

    The display code exists only in the return value; it is never stored.
    """

    code = generate_code()
    with cursor() as cur:
        cur.execute(
            """
            insert into access_codes (code_hash, label, is_admin, expires_at)
            values (%s, %s, %s, %s) returning id
            """,
            (hash_secret(normalize_code(code) or ""), label, is_admin, expires_at),
        )
        return code, str(cur.fetchone()["id"])


def list_codes() -> list[CodeRecord]:
    with cursor() as cur:
        cur.execute(
            """
            select c.*,
                   (select count(*) from videos v where v.owner_code_id = c.id) as video_count,
                   (select count(*) from sessions s
                     where s.code_id = c.id and s.expires_at > now()) as active_sessions
            from access_codes c
            order by c.created_at
            """
        )
        return [
            CodeRecord(
                id=str(row["id"]),
                label=row["label"],
                is_admin=row["is_admin"],
                created_at=row["created_at"],
                expires_at=row["expires_at"],
                revoked_at=row["revoked_at"],
                video_count=row["video_count"],
                active_sessions=row["active_sessions"],
            )
            for row in cur.fetchall()
        ]


def resolve_code_id(prefix: str) -> str:
    """Find the one code whose id starts with ``prefix``, as ``gist-access list`` prints it."""

    with cursor() as cur:
        cur.execute(
            "select id from access_codes where id::text like %s",
            (prefix.strip().lower() + "%",),
        )
        rows = cur.fetchall()
    if not rows:
        raise LookupError(f"no code id starts with {prefix!r}")
    if len(rows) > 1:
        raise LookupError(f"{prefix!r} matches {len(rows)} codes; give more of the id")
    return str(rows[0]["id"])


def revoke_code(code_id: str) -> None:
    """Revoke a code and end every session it holds. Its videos are kept."""

    with cursor() as cur:
        cur.execute(
            "update access_codes set revoked_at = coalesce(revoked_at, now()) where id = %s",
            (code_id,),
        )
        cur.execute("delete from sessions where code_id = %s", (code_id,))


def transfer_videos(from_code_id: str, to_code_id: str) -> int:
    """Move every video from one code to another, e.g. to replace a lost code.

    A video the receiving code already holds under the same link is left where
    it is rather than violating the one-copy-per-owner rule.
    """

    with cursor() as cur:
        cur.execute(
            """
            update videos v set owner_code_id = %(to)s
            where v.owner_code_id = %(from)s
              and not exists (
                  select 1 from videos o
                  where o.owner_code_id = %(to)s and o.youtube_url = v.youtube_url
              )
            """,
            {"from": from_code_id, "to": to_code_id},
        )
        return cur.rowcount


def claim_unowned_videos(code_id: str) -> int:
    """Give every video added before access codes existed to ``code_id``."""

    with cursor() as cur:
        cur.execute(
            "update videos set owner_code_id = %s where owner_code_id is null",
            (code_id,),
        )
        return cur.rowcount


# --------------------------------------------------------------- sessions ----


def login(code: str) -> tuple[str, Principal] | None:
    """Exchange a typed code for a new session token, or None if it is not valid."""

    normalized = normalize_code(code)
    if normalized is None:
        return None

    with cursor() as cur:
        cur.execute(
            """
            select id, label, is_admin from access_codes
            where code_hash = %s
              and revoked_at is null
              and (expires_at is null or expires_at > now())
            """,
            (hash_secret(normalized),),
        )
        row = cur.fetchone()
        if row is None:
            return None

        token = generate_session_token()
        cur.execute("delete from sessions where expires_at <= now()")
        cur.execute(
            "insert into sessions (code_id, token_hash, expires_at) values (%s, %s, now() + %s)",
            (row["id"], hash_secret(token), SESSION_LIFETIME),
        )
        return token, Principal(
            code_id=str(row["id"]), label=row["label"], is_admin=row["is_admin"]
        )


def principal_for_token(token: str) -> Principal | None:
    """The code behind a live session token, or None.

    A session stops working the moment its code is revoked or expires, not
    only when the session itself runs out.
    """

    with cursor() as cur:
        cur.execute(
            """
            select c.id, c.label, c.is_admin
            from sessions s join access_codes c on c.id = s.code_id
            where s.token_hash = %s
              and s.expires_at > now()
              and c.revoked_at is null
              and (c.expires_at is null or c.expires_at > now())
            """,
            (hash_secret(token),),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return Principal(code_id=str(row["id"]), label=row["label"], is_admin=row["is_admin"])


def end_all_sessions(code_id: str) -> int:
    """Log a code out on every device. The code itself stays valid."""

    with cursor() as cur:
        cur.execute("delete from sessions where code_id = %s", (code_id,))
        return cur.rowcount


# ------------------------------------------------------------ daily cap ----


def cap_day(now: datetime | None = None) -> date:
    """The day the cap is counted against, in the cap's time zone."""

    return (now or datetime.now(CAP_TIMEZONE)).astimezone(CAP_TIMEZONE).date()


def cap_resets_at(now: datetime | None = None) -> datetime:
    """When the current day's count starts again from zero."""

    tomorrow = cap_day(now) + timedelta(days=1)
    return datetime.combine(tomorrow, time.min, tzinfo=CAP_TIMEZONE)


def charge_question(code_id: str, *, day: date, cap: int) -> bool:
    """Count one question against today's cap. False when the cap is already used.

    A single statement, so two questions sent at once cannot both take the last
    slot.
    """

    with cursor() as cur:
        cur.execute(
            """
            insert into question_usage (code_id, day, count) values (%(code)s, %(day)s, 1)
            on conflict (code_id, day) do update
                set count = question_usage.count + 1
                where question_usage.count < %(cap)s
            returning count
            """,
            {"code": code_id, "day": day, "cap": cap},
        )
        return cur.fetchone() is not None


def refund_question(code_id: str, *, day: date) -> None:
    """Give a question back when it failed before producing an answer."""

    with cursor() as cur:
        cur.execute(
            """
            update question_usage set count = greatest(count - 1, 0)
            where code_id = %s and day = %s
            """,
            (code_id, day),
        )


def questions_used(code_id: str, *, day: date) -> int:
    with cursor() as cur:
        cur.execute(
            "select count from question_usage where code_id = %s and day = %s",
            (code_id, day),
        )
        row: dict[str, Any] | None = cur.fetchone()
    return row["count"] if row else 0
