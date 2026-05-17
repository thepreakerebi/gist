# Gist

Gist is an audio-visual context compression layer for video LLMs. The first implementation focuses on a clean, testable Gist-core runtime:

- query-aware candidate scoring
- joint visual/audio budget arbitration
- MMR-based temporal diversity
- deterministic compression presets
- API-ready response metadata for observability

The current code does not pretend to run CLIP, CLAP, Whisper, or an Omni-LLM yet. Those are model adapters that will plug into the candidate-generation layer. This keeps the compression logic independently testable before expensive model integration.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
uvicorn gist.api.app:create_app --factory --reload
```

For real media extraction, install FFmpeg:

```bash
brew install ffmpeg
```

## Example Request

```bash
curl -X POST http://127.0.0.1:8000/v1/compressions \
  -H "content-type: application/json" \
  -d '{
    "video_id": "demo-001",
    "query": "when does the speaker mention pricing?",
    "duration_seconds": 120,
    "preset": "balanced",
    "visual_candidates": [
      {"id": "v1", "timestamp_seconds": 10, "text": "title slide"},
      {"id": "v2", "timestamp_seconds": 48, "text": "speaker explains pricing tiers"}
    ],
    "audio_candidates": [
      {"id": "a1", "timestamp_seconds": 47, "text": "pricing starts at ten dollars"},
      {"id": "a2", "timestamp_seconds": 90, "text": "closing remarks"}
    ]
  }'
```

## Ingest a Local Video

```bash
curl -X POST http://127.0.0.1:8000/v1/ingestions \
  -H "content-type: application/json" \
  -d '{
    "video_path": "/absolute/path/to/video.mp4",
    "output_root": ".gist/ingestions",
    "sample_count": 128,
    "audio_window_seconds": 1.0
  }'
```

This endpoint is intended for local development and controlled backend usage. Public deployments should use upload/object-storage based ingestion instead of accepting arbitrary filesystem paths.

## Architecture

```text
raw video/audio
  -> media ingestion (ffprobe metadata, sampled frames, audio windows)
  -> candidate extraction adapters (CLIP/CLAP/Whisper later)
  -> Gist-core selector
  -> compressed timestamped evidence
  -> video LLM / omni-LLM gateway
```

## Current Milestones

- API and typed compression contracts
- deterministic Gist-core candidate selector
- compression presets and response metrics
- adapter interfaces for future model integrations
- FFmpeg-backed media ingestion utilities
- structured ingestion manifests with sampled frames and audio windows
- local ingestion API endpoint for development workflows

## Development Principles

- deterministic by default
- small, typed domain objects
- model adapters outside the core selector
- explicit metrics in every compression response
- tests around algorithmic behavior before GPU/model work
