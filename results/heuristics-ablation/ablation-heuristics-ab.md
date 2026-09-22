# Gist Ablation Report

- Cases: 35
- Candidate pool held fixed per case; only the selection input varies.
- Uniform sampling uses the same budget the full Gist run selected.

## How to read this

All four modes run the current code over an identical per-case candidate pool and are scored against the same reference terms and timestamp ranges. The **continuous averages below are the comparison of record**: they show, at matched token reduction, which selection strategy best recovers the answer-bearing moment. The reference terms/ranges were authored from the full-Gist artifacts, so they are if anything biased in full Gist's favour; the baselines are judged against the same targets.

The per-case pass/fail table is secondary. Its gates were pinned to the exact evidence the committed suite selected, so a mode that picks a different-but-valid moment (e.g. one of several identical recurring slides) can score high on the averages yet miss a strict gate.

## Mode comparison

| Mode | Pass rate | Answer recall | Evidence coverage | Evidence relevance | Timestamp hit | Grounded | Token reduction | Avg selected |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Gist (audio+visual) | 74% (26/35) | 0.83 | 0.84 | 0.89 | 0.86 | 0.89 | 99.83% | 1.29 |
| Full Gist, coverage heuristics OFF | 71% (25/35) | 0.80 | 0.81 | 0.87 | 0.83 | 0.91 | 99.82% | 1.29 |
| Visual-only retrieval | 46% (16/35) | 0.55 | 0.57 | 0.65 | 0.60 | 0.71 | 99.55% | 2.26 |
| Transcript-only retrieval | 29% (10/35) | 0.34 | 0.34 | 0.36 | 0.31 | 0.41 | 99.99% | 0.54 |
| Score top-k (relevance only) | 29% (10/35) | 0.40 | 0.43 | 0.51 | 0.40 | 0.97 | 99.75% | 1.29 |
| Split budget first, even halves | 29% (10/35) | 0.40 | 0.43 | 0.50 | 0.40 | 1.00 | 99.80% | 1.14 |
| Split budget first, intent-aware | 57% (20/35) | 0.70 | 0.72 | 0.72 | 0.60 | 1.00 | 99.86% | 1.14 |
| Split budget + separate scoring (diagnostic) | 57% (20/35) | 0.70 | 0.72 | 0.72 | 0.60 | 1.00 | 99.86% | 1.14 |
| Uniform sampling | 14% (5/35) | 0.27 | 0.32 | 0.52 | 0.24 | 0.91 | 99.78% | 1.29 |

## Per-case pass/fail

| Case | Category | Full Gist (audio+visual) | Full Gist, coverage heuristics OFF | Visual-only retrieval | Transcript-only retrieval | Score top-k (relevance only) | Split budget first, even halves | Split budget first, intent-aware | Split budget + separate scoring (diagnostic) | Uniform sampling |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| paul-graham-startup-killer | speech_semantic | fail | fail | fail | pass | fail | fail | pass | pass | fail |
| paul-graham-yc-support | global_summary | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| kinect-opening-title | visual_object_action | pass | pass | pass | fail | fail | fail | fail | fail | fail |
| kinect-person-demo | mixed_av | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| kinect-post-interface-title | temporal_before_after | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-01-opening-title | visual_object_action | pass | pass | pass | fail | fail | fail | fail | fail | pass |
| bio-motor-control-lecture-02-opening-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | pass |
| bio-motor-control-lecture-02-next-week | temporal_before_after | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-02-after-opening-title | temporal_before_after | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-01-global-summary | global_summary | pass | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-02-global-summary | global_summary | pass | pass | pass | fail | fail | fail | fail | fail | fail |
| paul-graham-startup-ideas-unconsciously | speech_semantic | fail | fail | fail | pass | fail | fail | pass | pass | fail |
| bio-motor-control-lecture-01-chapter-1-title | visual_object_action | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-02-conceptual-models-title | visual_object_action | pass | pass | pass | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-02-agenda-after-title | temporal_before_after | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-02-characterization-modelling-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| bio-motor-control-lecture-02-further-reading-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| bio-motor-control-lecture-02-legged-locomotion-nature-title | visual_object_action | pass | pass | pass | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-02-equations-motion-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| bio-motor-control-lecture-01-sense-think-act-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| bio-motor-control-lecture-01-behavior-based-robotics-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| bio-motor-control-lecture-01-biology-robotics-loop-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| bio-motor-control-lecture-02-conceptual-models-legged-locomotion-title | visual_object_action | pass | pass | pass | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-01-course-overview-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| bio-motor-control-lecture-01-objectives-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| bio-motor-control-lecture-01-course-evaluation-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | fail |
| night-of-the-living-dead-nasa-conference | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | fail |
| night-of-the-living-dead-killings-locations | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | fail |
| nasa-sts115-p3p4-truss-newest-element | speech_semantic | pass | pass | fail | fail | fail | fail | pass | pass | fail |
| nasa-sts115-solar-wing-dimensions | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | fail |
| yale-financial-markets-behavioral-finance | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | pass |
| history-augustus-birthplace-palatine | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | pass |
| medicine-anatomy-peripheral-nervous-system | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | fail |
| manufacturing-crayon-packing | speech_semantic | pass | pass | fail | pass | fail | fail | fail | fail | fail |
| art-history-high-renaissance-big-three | speech_semantic | pass | pass | fail | pass | fail | fail | pass | pass | pass |
