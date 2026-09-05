import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gist.api.demo import demo_router
from gist.api.library import library_router
from gist.api.routes import router
from gist.db import connection as db

# Load a repo-root .env if present. Real environment variables take precedence
# (load_dotenv does not override already-set vars), so shell/systemd config wins
# over the file in deployment.
load_dotenv()


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
    # Comma-separated allowed origins for the frontend (e.g. the Vercel URL).
    # Defaults to "*" so local dev and the demo work out of the box.
    origins = [
        origin.strip()
        for origin in os.getenv("GIST_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(demo_router)
    app.include_router(library_router)
    return app


app = create_app()

