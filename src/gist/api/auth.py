"""Signing in with an access code, signing out, and the session dependency.

The session travels in an ``HttpOnly`` cookie rather than an ``Authorization``
header because the browser loads source video, clips and frames through plain
``<video src>`` and ``<img src>`` elements, which cannot attach headers but do
send cookies. ``HttpOnly`` also keeps the token out of reach of page scripts.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from gist.access import store
from gist.access.ratelimit import GENERAL_LIMITS, LOGIN_LIMITS, Limit, limiter
from gist.access.store import Principal
from gist.db import connection as db

COOKIE_NAME = "gist_session"

auth_router = APIRouter(prefix="/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    code: str = Field(min_length=1, max_length=64)


def client_address(request: Request) -> str:
    """The caller's address.

    Behind a reverse proxy this is only the real client when uvicorn trusts the
    proxy's forwarding headers (``--proxy-headers --forwarded-allow-ips``);
    otherwise every request appears to come from the proxy and all clients
    share one login limit.
    """

    return request.client.host if request.client else "unknown"


def enforce(key: str, limits: tuple[Limit, ...], message: str) -> None:
    """Raise 429 with Retry-After when ``key`` is over any of ``limits``."""

    wait = limiter.hit(key, *limits)
    if wait is not None:
        raise HTTPException(
            status_code=429,
            detail=message,
            headers={"Retry-After": str(int(wait))},
        )


def _require_db() -> None:
    if not db.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Sign-in needs DATABASE_URL set to a Postgres connection string.",
        )


def current_principal(request: Request) -> Principal:
    """The access code behind the request's session cookie, or 401."""

    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Sign in with your access code.")
    _require_db()
    principal = store.principal_for_token(token)
    if principal is None:
        raise HTTPException(
            status_code=401, detail="Your session has ended. Sign in with your access code."
        )
    enforce(f"general:{principal.code_id}", GENERAL_LIMITS, "Too many requests. Slow down.")
    return principal


CurrentPrincipal = Annotated[Principal, Depends(current_principal)]


def _cookie_secure(request: Request) -> bool:
    setting = os.getenv("GIST_COOKIE_SECURE", "").strip().lower()
    if setting in {"1", "true", "yes"}:
        return True
    if setting in {"0", "false", "no"}:
        return False
    return request.url.scheme == "https"


def _set_session_cookie(request: Request, response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=int(store.SESSION_LIFETIME.total_seconds()),
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        path="/",
        domain=os.getenv("GIST_COOKIE_DOMAIN") or None,
    )


def _clear_session_cookie(request: Request, response: Response) -> None:
    response.delete_cookie(
        COOKIE_NAME,
        httponly=True,
        secure=_cookie_secure(request),
        samesite="lax",
        path="/",
        domain=os.getenv("GIST_COOKIE_DOMAIN") or None,
    )


def account_payload(principal: Principal) -> dict[str, Any]:
    day = store.cap_day()
    used = store.questions_used(principal.code_id, day=day)
    limit = None if principal.is_admin else store.DAILY_QUESTION_CAP
    return {
        "label": principal.label,
        "is_admin": principal.is_admin,
        "questions_used_today": used,
        "daily_question_limit": limit,
        "questions_left_today": None if limit is None else max(limit - used, 0),
        "limit_resets_at": store.cap_resets_at().isoformat(),
    }


@auth_router.post("/login")
def login(body: LoginRequest, request: Request, response: Response) -> dict[str, Any]:
    enforce(
        f"login:{client_address(request)}",
        LOGIN_LIMITS,
        "Too many sign-in attempts. Wait a minute and try again.",
    )
    _require_db()
    result = store.login(body.code)
    if result is None:
        raise HTTPException(status_code=401, detail="That code is not valid.")
    token, principal = result
    _set_session_cookie(request, response, token)
    return account_payload(principal)


@auth_router.post("/logout", status_code=204)
def logout(request: Request, response: Response) -> Response:
    """Sign the code out on every device it is signed in on.

    The code stays valid: signing in with it again restores its library.
    Succeeds without a session too, so a stale tab can always clear its cookie.
    """

    token = request.cookies.get(COOKIE_NAME)
    if token and db.is_configured():
        principal = store.principal_for_token(token)
        if principal is not None:
            store.end_all_sessions(principal.code_id)
    response.status_code = 204
    _clear_session_cookie(request, response)
    return response


@auth_router.get("/me")
def me(principal: CurrentPrincipal) -> dict[str, Any]:
    return account_payload(principal)
