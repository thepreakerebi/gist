# Local Quality Evaluation

Gist needs a local quality suite because manual report inspection does not scale. The goal is to track whether query-aware compression keeps the right context while reducing video-LLM input cost.

## Commands

Validate the dataset without running compression:

```bash
gist-quality-eval \
  --dataset data/eval/local-quality.jsonl \
  --check-only
```

Run the current curated suite:

```bash
gist-quality-eval \
  --dataset data/eval/local-quality.jsonl \
  --output reports/local-quality/quality.json \
  --markdown-output reports/local-quality/quality.md \
  --html-output reports/local-quality/quality.html
```

## Dataset Types

Use `compression_path` when replaying an existing run:

```json
{"id":"case-id","compression_path":".gist/runs/video/query/compression.json","expected_answer_terms":["research"],"expected_evidence_terms":["research"],"relevant_ranges":[{"start_seconds":370,"end_seconds":400}],"min_answer_term_recall":0.75,"min_evidence_relevance_rate":0.8,"min_token_reduction_percent":90.0}
```

Use `video_path` when running a fresh compression:

```json
{"id":"case-id","video_path":"/absolute/path/to/video.mp4","query":"What does the speaker say about pricing?","processing_mode":"auto","visual_scorer":"clip_scene","audio_scorer":"baseline","adaptive_budget":true,"decompose_query":true,"expected_answer_terms":["pricing"],"expected_evidence_terms":["pricing"],"relevant_ranges":[{"start_seconds":600,"end_seconds":640}],"min_answer_term_recall":0.75,"min_evidence_relevance_rate":0.8,"min_token_reduction_percent":90.0}
```

## Quality Gates

Alpha is good enough when the curated suite reaches:

- `30+` real long-video questions.
- `>= 80%` pass rate.
- `>= 0.75` average answer term recall.
- `>= 0.80` average evidence relevance rate.
- `>= 0.75` average timestamp hit rate.
- `>= 90%` average token reduction for long-video cases.
- No repeated manual failure pattern across more than `3` cases.

Beta is good enough when the curated suite reaches:

- `75+` real questions across interviews, lectures, demos, healthcare-style educational videos, and visually grounded clips.
- `>= 90%` pass rate.
- `>= 0.85` average answer term recall.
- `>= 0.90` average evidence relevance rate.
- `>= 0.85` average timestamp hit rate.
- `>= 95%` average token reduction.

## Case Selection

Add cases that cover different failure modes:

- Speech-only factual questions.
- Visual-only questions where transcript is insufficient.
- Mixed audio-visual questions.
- Temporal before/after questions.
- Global summary questions.
- OCR/text-on-screen questions.
- Long videos over one hour.

## Practical Workflow

1. Run Gist on a video and inspect the HTML report.
2. Identify the correct answer terms and the timestamp range that supports the answer.
3. Add one line to `data/eval/local-quality.jsonl`.
4. Run `gist-quality-eval --dataset data/eval/local-quality.jsonl --check-only`.
5. Run the full quality command.
6. Fix retrieval, evidence pruning, or answer grounding only when a failure repeats across cases.
