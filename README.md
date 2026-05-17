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

Optional CLIP visual scoring dependencies:

```bash
pip install -e ".[vision]"
```

Optional Whisper speech transcription dependencies:

```bash
pip install -e ".[audio]"
```

Optional CLAP sound-event scoring dependencies:

```bash
pip install -e ".[sound]"
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

## Compress a Local Video End-to-End

```bash
curl -X POST http://127.0.0.1:8000/v1/local-video-compressions \
  -H "content-type: application/json" \
  -d '{
    "video_path": "/absolute/path/to/video.mp4",
    "output_root": ".gist/ingestions",
    "query": "when does the speaker mention pricing?",
    "preset": "balanced",
    "visual_scorer": "baseline",
    "audio_scorer": "baseline",
    "sample_count": 128,
    "audio_window_seconds": 1.0
  }'
```

Use `"visual_scorer": "clip"` to score sampled frames with CLIP after installing the optional vision dependencies. The default `"baseline"` mode remains dependency-light and deterministic. CLAP and Whisper adapters will extend the audio side without changing the compression API contract.

Use `"audio_scorer": "whisper"` to transcribe extracted audio windows with Faster Whisper after installing the optional audio dependencies. The transcript becomes the audio candidate text, so Gist-core can rank speech windows by query relevance.

Use `"audio_scorer": "clap"` to score extracted audio windows against sound-event queries after installing the optional sound dependencies. This is intended for non-speech audio such as applause, alarms, engines, music, impact sounds, or environmental events.

## Caching

Local video compression writes reusable JSON cache artifacts under `output_root/cache`:

- ingestion manifests keyed by video path, frame sample count, and audio-window duration
- candidate sets keyed by video ID, query, visual scorer, and audio scorer

This prevents repeated FFmpeg extraction, CLIP scoring, and Whisper transcription for identical local compression requests.

## Explainability

Compression responses include evidence-level metadata for debugging and evaluation:

- `selection_rank`: order in which MMR selected the item
- `relevance_score`: raw query relevance score
- `normalized_score`: within-modality z-score used for fair visual/audio arbitration
- `mmr_score`: final relevance-diversity score at selection time
- `source_score_type`: `lexical_overlap` or `model_saliency`
- `reason`: short explanation for why the evidence item was selected

## Architecture

```text
raw video/audio
  -> media ingestion (ffprobe metadata, sampled frames, audio windows)
  -> baseline candidate generation (optional CLIP, Whisper, and CLAP adapters)
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
- one-call local video compression pipeline
- optional CLIP visual frame scoring adapter
- optional Whisper audio transcription adapter
- optional CLAP sound-event scoring adapter
- disk-backed ingestion and candidate caching
- explainable selected-evidence metadata

## Development Principles

- deterministic by default
- small, typed domain objects
- model adapters outside the core selector
- explicit metrics in every compression response
- tests around algorithmic behavior before GPU/model work
