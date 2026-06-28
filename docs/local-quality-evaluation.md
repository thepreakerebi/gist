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

Run the current long-video baseline suite:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --run-quality \
  --output-root .gist/long-video-suite/baseline \
  --output reports/long-video-suite/baseline-quality.json \
  --markdown-output reports/long-video-suite/baseline-quality.md \
  --html-output reports/long-video-suite/baseline-quality.html \
  --min-cases 11 \
  --min-distinct-videos 5 \
  --min-distinct-domains 4 \
  --min-cases-per-category 1 \
  --min-avg-token-reduction-percent 95 \
  --max-noisy-transcript-warning-rate 0.15 \
  --min-transcript-metadata-rate 0.05 \
  --min-answered-rate 0.9 \
  --max-avg-selected-evidence 4 \
  --min-quality-pass-rate 0.7
```

Current baseline as of June 28, 2026:

- `11` real long-video cases across `5` videos and `4` domains.
- `90.91%` quality pass rate.
- `99.87%` average token reduction.
- `9.09%` noisy transcript warning rate.
- `9.09%` transcript metadata coverage because most older artifacts predate transcript metadata.
- One known failing case: `bio-motor-control-lecture-01-global-summary`, due to weak grounding on noisy transcript evidence.

Run the future target gates separately:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --output reports/long-video-suite/target-readiness.json \
  --markdown-output reports/long-video-suite/target-readiness.md \
  --html-output reports/long-video-suite/target-readiness.html
```

The target gates are expected to fail until the curated suite reaches `30+` long-video cases with at least `3` per major query category and regenerated transcript metadata on most artifacts.

Draft a new quality case from an existing run:

```bash
gist-quality-eval \
  --draft-case-from .gist/runs/video-slug/query-slug/compression.json \
  --case-id video-slug-question-slug
```

The command prints one JSONL record plus comment notes. Review the inferred answer terms, evidence terms, and timestamp ranges before appending it to `data/eval/local-quality.jsonl`.

Draft cases for every existing run:

```bash
gist-quality-eval \
  --draft-cases-from-root .gist/runs \
  --draft-output reports/local-quality/drafted-cases.jsonl
```

This writes valid JSONL only. Review the output before appending it to the curated dataset.

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

## Failure Categories

The report labels failures so fixes target the right layer:

- `answer_grounding`: selected evidence may be useful, but the answer missed required terms.
- `evidence_retrieval`: selected evidence text did not cover the expected support terms.
- `temporal_localization`: selected clips missed the expected timestamp range.
- `compression_budget`: token reduction fell below the required threshold.
- `evidence_pruning`: too many final evidence clips survived.
- `modality_balance`: required audio or visual evidence was missing.

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
3. Draft a case with `gist-quality-eval --draft-case-from`.
4. Edit the drafted terms and timestamp ranges.
5. Add one line to `data/eval/local-quality.jsonl`.
6. Run `gist-quality-eval --dataset data/eval/local-quality.jsonl --check-only`.
7. Run the full quality command.
8. Fix retrieval, evidence pruning, or answer grounding only when a failure repeats across cases.
