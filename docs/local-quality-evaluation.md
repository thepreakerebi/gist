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
  --min-cases 13 \
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

Current baseline as of June 29, 2026:

- `14` real long-video cases across `5` videos and `4` domains.
- `100.00%` quality pass rate.
- `99.74%` average token reduction.
- `7.14%` noisy transcript warning rate.
- `21.43%` transcript metadata coverage because most older artifacts predate transcript metadata.
- Category coverage: speech `3`, visual `3`, temporal `3`, global `3`, mixed AV `2`.
- No known failing baseline cases. The remaining long-video warning is noisy transcript evidence on `bio-motor-control-lecture-01-global-summary`.

Audit existing long-video artifacts for safe dataset additions:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --audit-root .gist/runs \
  --audit-output reports/long-video-suite/artifact-audit.json \
  --min-cases 13 \
  --min-distinct-videos 5 \
  --min-distinct-domains 4 \
  --min-cases-per-category 1 \
  --min-avg-token-reduction-percent 95 \
  --max-noisy-transcript-warning-rate 0.15 \
  --min-transcript-metadata-rate 0.05 \
  --min-answered-rate 0.9 \
  --max-avg-selected-evidence 4
```

The current audit found `25` local artifacts: `13` already curated, `0` safe uncurated long-video candidates, and `12` rejected because they are short, low-reduction, answer-noisy, or explicitly unreliable. Dataset expansion now requires new real long-video runs rather than recycling existing artifacts.

Run the future target gates separately:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --output reports/long-video-suite/target-readiness.json \
  --markdown-output reports/long-video-suite/target-readiness.md \
  --html-output reports/long-video-suite/target-readiness.html
```

The target gates are expected to fail until the curated suite reaches `30+` long-video cases with at least `3` per major query category and regenerated transcript metadata on most artifacts. The readiness report includes an `Expansion Plan` section that converts those failed gates into concrete curation targets, such as how many more long-video cases, distinct videos, domains, or query-category examples are needed before the suite is alpha-ready. It also includes `Query Proposals` that suggest category-specific questions against existing long-video sources; treat these as starting prompts, then verify the answer terms, evidence terms, and timestamp ranges from the generated report before adding a case to the curated dataset.

Write the combined roundup report:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --roundup-output reports/long-video-suite/roundup.json \
  --roundup-markdown-output reports/long-video-suite/roundup.md
```

Use this as the main handoff/status artifact. It combines target readiness,
curation gaps, transcript metadata refresh gaps, next curation command, next
metadata refresh command, and the promotion command template. When `Ready for
paper freeze` is `yes`, the curated long-video suite is ready to freeze for the
research paper experiments.

Write the next-action curation queue:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --queue-output reports/long-video-suite/curation-queue.json \
  --queue-markdown-output reports/long-video-suite/curation-queue.md
```

The queue includes current progress, missing category coverage, priority actions,
and copy-ready `--curate-proposal-index` commands for the next proposed cases.

Write the transcript metadata refresh queue:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --metadata-refresh-output reports/long-video-suite/metadata-refresh-queue.json \
  --metadata-refresh-markdown-output reports/long-video-suite/metadata-refresh-queue.md \
  --metadata-refresh-output-root .gist/metadata-refresh \
  --metadata-refresh-quality balanced \
  --metadata-refresh-visual-scorer baseline
```

Use this when target readiness fails on `transcript_metadata_rate`. The generated
commands rerun older curated cases with Whisper-backed audio scoring so refreshed
artifacts include transcript metadata. The default refresh visual scorer is
`baseline` to avoid unnecessary CLIP downloads during transcript-only refreshes.
Refreshed artifacts are written to `.gist/metadata-refresh` by default so curated
dataset artifacts are not overwritten before review.

Run a controlled metadata refresh batch:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --run-metadata-refresh \
  --metadata-refresh-limit 1 \
  --metadata-refresh-output-root .gist/metadata-refresh \
  --metadata-refresh-run-output reports/long-video-suite/metadata-refresh-run.json
```

Start with a limit of `1`, inspect the regenerated report, then only promote or
recapture the case if the refreshed artifact still passes the curated quality
thresholds.

Promote a reviewed metadata refresh artifact back to its curated run path:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --promote-metadata-refresh-case paul-graham-startup-killer \
  --promote-metadata-refresh-compression .gist/metadata-refresh/video-slug/query-slug/compression.json \
  --metadata-refresh-promotion-mode metadata-only \
  --metadata-refresh-promotion-output reports/long-video-suite/metadata-refresh-promotion.json
```

Promotion is gated by the existing quality thresholds for that case. If the
refreshed artifact fails in `full` mode, the curated artifact is left untouched.
Use `metadata-only` mode for transcript coverage cleanup: it validates that the
existing curated artifact still passes, then copies only `transcript_metadata`
from the refreshed artifact so verified answer/evidence clips are preserved.

Run a proposed query and write a review bundle:

```bash
gist-long-video-suite \
  --dataset data/eval/long-video-quality.jsonl \
  --curate-proposal-index 0 \
  --curation-output-root .gist/curation \
  --curation-visual-scorer clip_scene \
  --curation-audio-scorer whisper \
  --curation-audio-window-seconds 30 \
  --curation-whisper-max-windows 3
```

The curation command writes `compression.json`, `report.html`, and
`quality-case.draft.jsonl` for the selected proposal. It returns success when the
review bundle is written, even if the readiness gates still fail. Review the HTML
report and edit the drafted answer terms, evidence terms, grounding threshold, and
timestamp ranges before appending the case to `data/eval/long-video-quality.jsonl`.
Whisper is the default curation audio scorer because quality-case drafts need
semantic transcript evidence. The curation default transcribes at most three
evenly distributed 30-second windows to keep 1-hour videos practical on a local
machine. This is a fast draft mode; manually review the report and rerun with a
higher `--curation-whisper-max-windows` value when a candidate needs denser
transcript coverage before promotion. Use `baseline` only for dependency-light
smoke runs that will not be promoted into the curated dataset.

Validate the edited draft before appending it:

```bash
gist-long-video-suite \
  --review-draft .gist/curation/video-slug/query-slug/quality-case.draft.jsonl \
  --review-markdown-output .gist/curation/video-slug/query-slug/curation-review.md
```

The review command returns `ready_for_dataset=yes` only when the draft has the
required metadata, verified terms, timestamp ranges, grounding threshold, token
reduction target, evidence cap, and mixed audio/visual evidence floors where
applicable.

Once the edited draft passes, append it through the same gate:

```bash
gist-long-video-suite \
  --review-draft .gist/curation/video-slug/query-slug/quality-case.draft.jsonl \
  --append-draft-to data/eval/long-video-quality.jsonl
```

This command refuses to append drafts that fail readiness checks or reuse an
existing case id.

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
