import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gist.api.auth import auth_router
from gist.api.demo import demo_router
from gist.api.library import library_router
from gist.api.routes import research_router, router
from gist.db import connection as db

# Load a repo-root .env if present. Real environment variables take precedence
# (load_dotenv does not override already-set vars), so shell/systemd config wins
# over the file in deployment.
load_dotenv()

# The web app in local development. A deployment sets GIST_CORS_ORIGINS to the
# frontend's own origin.
DEFAULT_ORIGINS = ["http://localhost:3000"]
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def allowed_origins() -> list[str]:
    """Origins the browser may call the API from, with the session cookie.

    A wildcard is refused rather than honoured: with credentials allowed,
    Starlette answers "*" by echoing whatever origin asked, which would let any
    website act as a signed-in visitor.
    """

    configured = os.getenv("GIST_CORS_ORIGINS", "")
    origins = [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]
    if "*" in origins:
        raise ValueError(
            "GIST_CORS_ORIGINS cannot be '*' now that the API uses a session cookie; "
            "list the frontend's origin, e.g. https://gist.example.com"
        )
    return origins or DEFAULT_ORIGINS


def research_routes_enabled() -> bool:
    return os.getenv("GIST_ENABLE_RESEARCH_ROUTES", "").strip().lower() in {"1", "true", "yes"}


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Idempotent DDL on boot. The library degrades to a 503 rather than taking
    # the whole API down when no DATABASE_URL is configured, so this is
    # best-effort: compression endpoints must work without a database.
    if db.is_configured():
        try:
            db.apply_schema()
        except Exception:  # noqa: BLE001
            pass
    yield
    db.close_pool()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gist API",
        version="0.1.0",
        description="Audio-visual context compression API for video LLM pipelines.",
        lifespan=_lifespan,
    )
    origins = allowed_origins()

    @app.middleware("http")
    async def reject_foreign_origins(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # A state-changing request from a page on another site is refused even
        # before the cookie's SameSite rule or CORS come into play. Requests
        # without an Origin header (curl, server-to-server) are not browsers
        # acting on someone's behalf and pass through to normal authentication.
        origin = request.headers.get("origin")
        if request.method in _UNSAFE_METHODS and origin and origin.rstrip("/") not in origins:
            return JSONResponse({"detail": "origin not allowed"}, status_code=403)
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )
    app.include_router(router)
    app.include_router(auth_router)
    app.include_router(library_router)
    if research_routes_enabled():
        app.include_router(research_router)
        app.include_router(demo_router)
    return app


app = create_app()
