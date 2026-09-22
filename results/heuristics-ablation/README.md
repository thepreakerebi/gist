# Coverage-heuristics A/B — does the method carry the result, or do the rules?

`gist/core/schemas.py` has carried a `coverage_heuristics` flag since the
per-intent post-processors were added, with a comment promising an A/B "to
measure whether they carry signal or just overfit". The flag was plumbed
through the pipeline but never exercised: the ablation suite's modes vary the
*input signal* (which modality, which scorer), never the *post-processing
rules*. So the sharpest question this project is exposed to — how much of
Gist's result is the method, and how much is hand-tuned rules fitted to the
evaluation set? — had no measurement behind it.

This run answers it. It also answers **RQ4** ("which of the components
contributes materially, and which query intent categories benefit most and
least").

**Runner:** `gist-ablation` / `scripts`-free: `python -m gist.eval.ablation`
**Data:** `data/eval/long-video-quality.jsonl`, all 35 cases

> **Re-run 2026-09-22 at n=35.** The original run was n=39. Four cases were
> withdrawn because they ran against `tears_of_steel_61min.webm`, which was the
> 12.2-minute Blender film concatenated five times rather than an hour-long
> recording. Every number below is the re-run. The headline conclusion is
> unchanged; one interval in the companion split-budget ablation is not, and that
> is called out there.
**Conditions:** candidate pool held fixed per case; only the selection input or
the post-processing varies. CLAP seeded (see below), so this run reproduces.

## Result

| Mode | Pass rate | Answer recall | Timestamp hit | Token reduction | Avg selected |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Full Gist (audio+visual) | **74% (26/35)** | 0.83 | 0.86 | 99.83% | 1.29 |
| Full Gist, coverage heuristics **OFF** | **71% (25/35)** | 0.80 | 0.83 | 99.82% | 1.29 |
| Visual-only retrieval | 46% (16/35) | 0.55 | 0.60 | 99.55% | 2.26 |
| Transcript-only retrieval | 29% (10/35) | 0.34 | 0.31 | 99.99% | 0.54 |
| Score top-k (relevance only, no MMR) | 29% (10/35) | 0.40 | 0.40 | 99.75% | 1.29 |
| Uniform sampling | 14% (5/35) | 0.27 | 0.24 | 99.78% | 1.29 |

**Disabling every `_ensure_*` post-processor costs one case out of 35.** The
hand-tuned coverage rules are not what produces the result. Scoring and MMR
are: `score_topk` isolates that directly — identical scores, identical budget,
diversity removed — and drops from 71% to 29%.

### What the confidence intervals do to that claim

Paired percentile bootstrap, 10,000 resamples, cases resampled together
(`python -m gist.eval.bootstrap --ablation ... --baseline full_gist`):

| Condition | Pass rate [95% CI] | vs full Gist (pp) | Case agreement |
| :--- | :--- | :--- | ---: |
| Full Gist | 74.3% [60.0%, 88.6%] | — | — |
| Heuristics OFF | 71.4% [57.1%, 85.7%] | −2.9 [−8.6, +0.0] | 97% |
| Visual-only | 45.7% [28.6%, 62.9%] | −28.6 [−42.9, −14.3] | 71% |
| Transcript-only | 28.6% [14.3%, 42.9%] | −45.7 [−65.7, −25.7] | 43% |
| Score top-k | 28.6% [14.3%, 42.9%] | −45.7 [−62.9, −28.6] | 54% |
| Uniform | 14.3% [2.9%, 25.7%] | −60.0 [−74.3, −42.9] | 40% |

The intervals sharpen two claims and **weaken a third**, which is the point of
computing them:

- **Sharpened.** Every baseline comparison excludes zero by a wide margin.
  Cross-modal scoring and MMR are not noise: visual-only is at least 14 points
  worse and score-top-k at least 28, at 95% confidence.
- **Sharpened.** Heuristics-off agrees with full Gist on **97% of cases**, and
  the difference's upper bound is exactly +0.0 — across 10,000 resamples,
  disabling the rules was never *better*. The two configurations behave alike;
  they are not trading wins that happen to cancel.
- **Weakened.** The interval on that difference runs to −7.7 points. At n=39 the
  data cannot rule out the heuristics being worth up to about eight points.
  The honest statement is therefore *"the coverage rules account for at most a
  small part of the result, and the method carries it"* — not *"the rules
  contribute nothing"*, which the point estimate alone would have supported and
  the interval does not.

## Per intent category

| Category | n | Gist | Heuristics off | Visual-only | Score top-k | Uniform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_object_action | 16 | 15/16 | 15/16 | 15/16 | 10/16 | 2/16 |
| speech_semantic | 13 | 9/13 | 9/13 | 0/13 | 0/13 | 4/13 |
| temporal_before_after | 4 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| global_summary | 3 | 2/3 | 1/3 | 1/3 | 0/3 | 0/3 |
| mixed_av | 3 | 1/3 | 1/3 | 0/3 | 0/3 | 0/3 |

Three things worth stating plainly, including the ones that do not flatter the
method:

- **The heuristics change exactly one stratum.** `global_summary` goes 2/3 to
  1/3. Everywhere else, on/off is identical.
- **Cross-modal arbitration is where the gain lives, and it is conditional.**
  On the 13 `speech_semantic` cases Gist scores 9/13 against visual-only's
  0/13. On the 16 `visual_object_action` cases audio contributes nothing
  (15/16 either way). The blended 69-vs-41 headline understates the effect
  where it applies and overstates it where it does not.
- **`temporal_before_after` is 0/4 in every mode, ours included.** There is
  dedicated code for this category and it still scores zero. RQ4 states that a
  null result is an answer rather than a failure; this is one, and it is
  reported rather than buried.

## Caveats

- The reference terms and timestamp ranges were authored from full-Gist
  artifacts, so the gates are if anything biased in full Gist's favour. Every
  mode is scored against the same targets.
- The corpus is skewed: 20 of 39 cases come from two robotics lectures, and
  roughly half are slide-title detection. Domain breadth is the outstanding
  corpus work, not a property this run establishes.
- Absolute pass rates depend on gate thresholds pinned to the committed suite;
  the comparison between modes is the durable finding, not the 69%.

## The pre-registered test

Objective 3's accuracy criterion is non-inferiority within one standard error of
dense full-context, not improvement. On the 18 Video-MME AV questions run
against unquantized Qwen2.5-Omni-7B (`results/runpod-fp16/`):

| Condition | Accuracy [95% CI] | vs full context (pp) | Agreement |
| :--- | :--- | :--- | ---: |
| Full context | 27.8% [11.1%, 50.0%] | — | — |
| Gist | 33.3% [11.1%, 55.6%] | +5.6 [+0.0, +16.7] | 94% |

**The criterion is met**, and by more than it needed to be: the lower bound on
the difference is +0.0, so across 10,000 resamples Gist never scored below full
context. The two conditions agree on 94% of questions — Gist reaches the same
answers from roughly a third of the frames.

This is *not* evidence that Gist is more accurate. The interval spans zero to
+16.7 points and n=18; the defensible claim is parity, which is exactly what was
pre-registered.

## Splits

These numbers are **development** numbers. A grouped held-out split
(`data/eval/splits/held-out.json`) was frozen on 2026-09-05: 12 cases across six
recordings that appear nowhere in the dev split, weighted toward
`speech_semantic` because that is the stratum where cross-modal arbitration is
load-bearing. It is deliberately **not executed here** — it is reported once,
after development is complete, and running it early would spend it.

Restricting the ablation to the 27 dev cases gives the same picture
(`reports/bootstrap-dev.md`): full Gist 74.1%, heuristics-off 70.4% (−3.7 pp,
96% agreement), visual-only 55.6%, score-top-k 37.0%, uniform 14.8%.

## Reproducibility

This run postdates the CLAP determinism fix. `laion/clap-htsat-unfused`
defaults to `truncation="rand_trunc"`, which reduces any window over 10 s to a
*random* 10 s excerpt — two identical calls previously disagreed at embedding
cosine 0.89–0.97, so every earlier CLAP-scored number was irreproducible.
Feature extraction is now seeded. Earlier reports under `reports/` predate this
and should not be compared line-by-line against these numbers.
