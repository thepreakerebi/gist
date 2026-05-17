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

## Development Principles

- deterministic by default
- small, typed domain objects
- model adapters outside the core selector
- explicit metrics in every compression response
- tests around algorithmic behavior before GPU/model work
