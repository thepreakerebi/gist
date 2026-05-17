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

Run slow model adapter tests only after installing optional dependencies:

```bash
GIST_RUN_SLOW_MODEL_TESTS=1 python -m pytest -m slow
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
    "adaptive_budget": false,
    "decompose_query": false,
    "sample_count": 128,
    "audio_window_seconds": 1.0
  }'
```

Use `"visual_scorer": "clip"` to score sampled frames with CLIP after installing the optional vision dependencies. The default `"baseline"` mode remains dependency-light and deterministic. CLAP and Whisper adapters will extend the audio side without changing the compression API contract.

Use `"audio_scorer": "whisper"` to transcribe extracted audio windows with Faster Whisper after installing the optional audio dependencies. The transcript becomes the audio candidate text, so Gist-core can rank speech windows by query relevance.

Use `"audio_scorer": "clap"` to score extracted audio windows against sound-event queries after installing the optional sound dependencies. This is intended for non-speech audio such as applause, alarms, engines, music, impact sounds, or environmental events.

Use `"decompose_query": true` to split compound questions into independently scoreable aspects before compression. The current decomposer is deterministic and rule-based; an LLM decomposer can replace it later without changing the response shape.

Use `"adaptive_budget": true` to start with the aggressive preset and automatically expand to a larger budget when the aggressive evidence looks weak or one-sided.

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
- `query_aspects`: decomposed query aspects used for scoring when enabled
- `budget_mode`, `budget_preset_used`, and `budget_expanded`: adaptive-budget routing metadata

## Evaluation

Run the lightweight JSONL evaluation harness:

```bash
gist-eval \
  --dataset data/eval/demo.jsonl \
  --output reports/eval.json \
  --output-root .gist/eval \
  --markdown-output reports/eval.md \
  --html-output reports/eval.html \
  --preset aggressive
```

By default, `gist-eval` compares several variants:

- `gist_fixed_balanced`
- `gist_fixed_aggressive`
- `gist_decomposed`
- `gist_adaptive`
- `gist_decomposed_adaptive`

Use `--single-config` with `--preset`, `--decompose-query`, and `--adaptive-budget` to run only one configured variant.

The dataset format is one JSON object per line with:

- `id`
- `video_id`
- `query`
- `duration_seconds`
- `relevant_timestamps`
- `timestamp_tolerance_seconds`
- `visual_candidates`
- `audio_candidates`

For real-video evaluation, provide `video_path` instead of prebuilt candidates:

```json
{"id":"case-1","video_id":"demo","video_path":"/absolute/path/video.mp4","query":"when does applause happen","duration_seconds":120,"relevant_timestamps":[42],"sample_count":128,"audio_window_seconds":1.0}
```

Real-video examples use the local video pipeline, so `--output-root` stores extracted frames, audio windows, and reusable cache artifacts.

The report compares Gist variants against a uniform timestamp baseline and includes reduction percent, timestamp hit rate, modality coverage, and latency.
Use `--html-output` for a self-contained evidence inspection report.

## Token Estimates

Gist reports proxy token savings in compression and evaluation outputs:

- visual candidate: `258` estimated tokens
- audio candidate: `32` estimated tokens

These defaults are conservative model-agnostic proxies inspired by current video-LLM pricing/tokenization patterns. They are not provider billing guarantees. Provider-specific token estimators can replace this layer later.

Available token estimator profiles:

- `generic`
- `gemini_default`
- `gemini_low_res`

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
- rule-based query decomposition for compound questions
- adaptive budget routing for difficult queries
- JSONL evaluation harness with multi-variant comparison
- real-video evaluation through the local ingestion pipeline
- proxy token-savings estimates for compression and evaluation reports
- gated slow integration tests for CLIP, Whisper, and CLAP adapters
- HTML evidence reports for evaluation output
- provider-neutral downstream LLM gateway interface

## Development Principles

- deterministic by default
- small, typed domain objects
- model adapters outside the core selector
- explicit metrics in every compression response
- tests around algorithmic behavior before GPU/model work
