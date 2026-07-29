# Hugging Face Space (Docker SDK, free CPU Basic) for the Gist live demo API.
#
# Runs the full Gist pipeline on CPU and answers via a hosted multimodal LLM
# (OpenAI/Claude) selected per-request. The hosted answerer is for demo
# RELIABILITY only — the capstone paper's efficiency/FLOP claims come from
# Qwen2.5-Omni-7B measured offline, never from this API.
#
# Set as Space secrets: OPENAI_API_KEY, ANTHROPIC_API_KEY.
# Optionally set GIST_CORS_ORIGINS to your Vercel origin (defaults to "*").

FROM python:3.11-slim

# ffmpeg/ffprobe: frame extraction + duration probe. git: pip VCS installs if any.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run as a non-root user with UID 1000; model caches must be writable.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false \
    PORT=7860

USER user
WORKDIR /home/user/app

# Install deps first for layer caching. Torch CPU wheels keep the image lean.
COPY --chown=user pyproject.toml README.md ./
COPY --chown=user src ./src
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[vision,audio,sound]" "yt-dlp>=2025.1.0"

EXPOSE 7860

CMD ["uvicorn", "gist.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
