# Gist Ablation Report

- Cases: 39
- Candidate pool held fixed per case; only the selection input varies.
- Uniform sampling uses the same budget the full Gist run selected.

## How to read this

All four modes run the current code over an identical per-case candidate pool and are scored against the same reference terms and timestamp ranges. The **continuous averages below are the comparison of record**: they show, at matched token reduction, which selection strategy best recovers the answer-bearing moment. The reference terms/ranges were authored from the full-Gist artifacts, so they are if anything biased in full Gist's favour; the baselines are judged against the same targets.

The per-case pass/fail table is secondary. Its gates were pinned to the exact evidence the committed suite selected, so a mode that picks a different-but-valid moment (e.g. one of several identical recurring slides) can score high on the averages yet miss a strict gate.

## Mode comparison

| Mode | Pass rate | Answer recall | Evidence coverage | Evidence relevance | Timestamp hit | Grounded | Token reduction | Avg selected |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Gist (audio+visual) | 69% (27/39) | 0.81 | 0.83 | 0.86 | 0.82 | 0.90 | 99.84% | 1.31 |
| Full Gist, coverage heuristics OFF | 67% (26/39) | 0.79 | 0.80 | 0.86 | 0.79 | 0.92 | 99.84% | 1.28 |
| Visual-only retrieval | 41% (16/39) | 0.50 | 0.51 | 0.59 | 0.54 | 0.67 | 99.52% | 2.41 |
| Transcript-only retrieval | 26% (10/39) | 0.37 | 0.38 | 0.40 | 0.33 | 0.47 | 99.98% | 0.64 |
| Score top-k (relevance only) | 26% (10/39) | 0.39 | 0.41 | 0.48 | 0.38 | 0.97 | 99.75% | 1.31 |
| Split budget first, even halves | 26% (10/39) | 0.37 | 0.39 | 0.46 | 0.36 | 1.00 | 99.80% | 1.15 |
| Split budget first, intent-aware | 51% (20/39) | 0.66 | 0.68 | 0.69 | 0.54 | 1.00 | 99.86% | 1.15 |
| Split budget + separate scoring (diagnostic) | 51% (20/39) | 0.66 | 0.68 | 0.69 | 0.54 | 1.00 | 99.86% | 1.15 |
| Uniform sampling | 15% (6/39) | 0.30 | 0.34 | 0.52 | 0.27 | 0.92 | 99.79% | 1.31 |

## Per-case pass/fail

| Case | Category | Full Gist (audio+visual) | Full Gist, coverage heuristics OFF | Visual-only retrieval | Transcript-only retrieval | Score top-k (relevance only) | Split budget first, even halves | Split budget first, intent-aware | Split budget + separate scoring (diagnostic) | Uniform sampling |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| paul-graham-startup-killer | speech_semantic | fail | fail | fail | pass | fail | fail | pass | pass | fail |
| paul-graham-yc-support | global_summary | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| tears-of-steel-robot-hand-fear | speech_semantic | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| tears-of-steel-robotics-space | speech_semantic | fail | fail | fail | fail | fail | fail | fail | fail | pass |
| kinect-opening-title | visual_object_action | pass | pass | pass | fail | fail | fail | fail | fail | fail |
| kinect-person-demo | mixed_av | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| kinect-post-interface-title | temporal_before_after | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-01-opening-title | visual_object_action | pass | pass | pass | fail | fail | fail | fail | fail | pass |
| bio-motor-control-lecture-02-opening-title | visual_object_action | pass | pass | pass | fail | pass | pass | pass | pass | pass |
| bio-motor-control-lecture-02-next-week | temporal_before_after | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-02-after-opening-title | temporal_before_after | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-01-global-summary | global_summary | pass | fail | fail | fail | fail | fail | fail | fail | fail |
| bio-motor-control-lecture-02-global-summary | global_summary | pass | pass | pass | fail | fail | fail | fail | fail | fail |
| tears-of-steel-robotics-space-robot-hand | mixed_av | fail | fail | fail | fail | fail | fail | fail | fail | fail |
| tears-of-steel-robot-hand-visible-question | mixed_av | pass | pass | fail | fail | fail | fail | fail | fail | fail |
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
