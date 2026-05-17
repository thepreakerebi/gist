from fastapi import FastAPI

from gist.api.routes import router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gist API",
        version="0.1.0",
        description="Audio-visual context compression API for video LLM pipelines.",
    )
    app.include_router(router)
    return app


app = create_app()

