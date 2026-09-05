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
**Data:** `data/eval/long-video-quality.jsonl`, all 39 cases
**Conditions:** candidate pool held fixed per case; only the selection input or
the post-processing varies. CLAP seeded (see below), so this run reproduces.

## Result

| Mode | Pass rate | Answer recall | Timestamp hit | Token reduction | Avg selected |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Full Gist (audio+visual) | **69% (27/39)** | 0.81 | 0.82 | 99.84% | 1.31 |
| Full Gist, coverage heuristics **OFF** | **67% (26/39)** | 0.79 | 0.79 | 99.84% | 1.28 |
| Visual-only retrieval | 41% (16/39) | 0.50 | 0.54 | 99.52% | 2.41 |
| Transcript-only retrieval | 26% (10/39) | 0.37 | 0.33 | 99.98% | 0.64 |
| Score top-k (relevance only, no MMR) | 26% (10/39) | 0.39 | 0.38 | 99.75% | 1.31 |
| Uniform sampling | 15% (6/39) | 0.30 | 0.27 | 99.79% | 1.31 |

**Disabling every `_ensure_*` post-processor costs one case out of 39.** The
hand-tuned coverage rules are not what produces the result. Scoring and MMR
are: `score_topk` isolates that directly — identical scores, identical budget,
diversity removed — and drops from 67% to 26%.

### What the confidence intervals do to that claim

Paired percentile bootstrap, 10,000 resamples, cases resampled together
(`python -m gist.eval.bootstrap --ablation ... --baseline full_gist`):

| Condition | Pass rate [95% CI] | vs full Gist (pp) | Case agreement |
| :--- | :--- | :--- | ---: |
| Full Gist | 69.2% [53.8%, 84.6%] | — | — |
| Heuristics OFF | 66.7% [51.3%, 82.1%] | −2.6 [−7.7, +0.0] | 97% |
| Visual-only | 41.0% [25.6%, 56.4%] | −28.2 [−43.6, −15.4] | 72% |
| Transcript-only | 25.6% [12.8%, 38.5%] | −43.6 [−61.5, −25.6] | 46% |
| Score top-k | 25.6% [12.8%, 41.0%] | −43.6 [−59.0, −28.2] | 56% |
| Uniform | 15.4% [5.1%, 28.2%] | −53.8 [−69.2, −35.9] | 41% |

The intervals sharpen two claims and **weaken a third**, which is the point of
computing them:

- **Sharpened.** Every baseline comparison excludes zero by a wide margin.
  Cross-modal scoring and MMR are not noise: visual-only is at least 15 points
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
