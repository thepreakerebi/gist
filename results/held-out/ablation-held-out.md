# Gist Ablation Report

- Cases: 12
- Candidate pool held fixed per case; only the selection input varies.
- Uniform sampling uses the same budget the full Gist run selected.

## How to read this

All four modes run the current code over an identical per-case candidate pool and are scored against the same reference terms and timestamp ranges. The **continuous averages below are the comparison of record**: they show, at matched token reduction, which selection strategy best recovers the answer-bearing moment. The reference terms/ranges were authored from the full-Gist artifacts, so they are if anything biased in full Gist's favour; the baselines are judged against the same targets.

The per-case pass/fail table is secondary. Its gates were pinned to the exact evidence the committed suite selected, so a mode that picks a different-but-valid moment (e.g. one of several identical recurring slides) can score high on the averages yet miss a strict gate.

## Mode comparison

| Mode | Pass rate | Answer recall | Evidence coverage | Evidence relevance | Timestamp hit | Grounded | Token reduction | Avg selected |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Gist (audio+visual) | 58% (7/12) | 0.67 | 0.67 | 0.88 | 0.83 | 0.75 | 99.89% | 1.08 |
| Full Gist, coverage heuristics OFF | 58% (7/12) | 0.67 | 0.67 | 0.88 | 0.83 | 0.75 | 99.89% | 1.08 |
| Visual-only retrieval | 8% (1/12) | 0.17 | 0.19 | 0.47 | 0.42 | 0.42 | 99.41% | 2.67 |
| Transcript-only retrieval | 58% (7/12) | 0.66 | 0.66 | 0.75 | 0.58 | 0.83 | 99.98% | 0.83 |
| Score top-k (relevance only) | 0% (0/12) | 0.00 | 0.00 | 0.25 | 0.25 | 0.92 | 99.77% | 1.08 |
| Split budget first, even halves | 0% (0/12) | 0.00 | 0.00 | 0.25 | 0.25 | 1.00 | 99.79% | 1.00 |
| Split budget first, intent-aware | 67% (8/12) | 0.69 | 0.69 | 0.75 | 0.67 | 1.00 | 99.93% | 1.00 |
| Split budget + separate scoring (diagnostic) | 67% (8/12) | 0.69 | 0.69 | 0.75 | 0.67 | 1.00 | 99.93% | 1.00 |
| Uniform sampling | 17% (2/12) | 0.22 | 0.22 | 0.50 | 0.42 | 0.75 | 99.83% | 1.08 |

## Per-case pass/fail

| Case | Category | Full Gist (audio+visual) | Full Gist, coverage heuristics OFF | Visual-only retrieval | Transcript-only retrieval | Score top-k (relevance only) | Split budget first, even halves | Split budget first, intent-aware | Split budget + separate scoring (diagnostic) | Uniform sampling |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| paul-graham-startup-killer | speech_semantic | fail | fail | fail | pass | fail | fail | pass | pass | fail |
| paul-graham-yc-support | global_summary | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| kinect-opening-title | visual_object_action | pass | pass | pass | fail | fail | fail | fail | fail | fail |
| kinect-person-demo | mixed_av | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| kinect-post-interface-title | temporal_before_after | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| paul-graham-startup-ideas-unconsciously | speech_semantic | fail | fail | fail | pass | fail | fail | pass | pass | fail |
| night-of-the-living-dead-nasa-conference | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | fail |
| night-of-the-living-dead-killings-locations | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | fail |
| nasa-sts115-p3p4-truss-newest-element | speech_semantic | pass | pass | fail | fail | fail | fail | pass | pass | fail |
| nasa-sts115-solar-wing-dimensions | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | fail |
| history-augustus-birthplace-palatine | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | pass |
| art-history-high-renaissance-big-three | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | pass |
