import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from gist.api.demo import demo_router
from gist.api.routes import router

# Load a repo-root .env if present. Real environment variables take precedence
# (load_dotenv does not override already-set vars), so shell/systemd config wins
# over the file in deployment.
load_dotenv()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gist API",
        version="0.1.0",
        description="Audio-visual context compression API for video LLM pipelines.",
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
    return app


app = create_app()

