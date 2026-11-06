# Held-out split: the one execution

The frozen held-out split (`data/eval/splits/held-out.json`, frozen 2026-09-05)
run once, on 2026-11-06, at commit `da0661e`. Twelve cases over six recordings
that appear nowhere in the dev split, so no part of Gist was tuned on them. The
measure is the same as the dev ablation in `../heuristics-ablation/`: whether the
selected evidence passes each case's thresholds (answer and evidence terms,
relevance, reference timestamp window, grounding, budget). No answering model is
involved, and the run used a laptop CPU.

## Checks before the run

- The corpus fingerprint and the held-out fingerprint both matched the frozen
  manifest, so the split had not changed since it was frozen.
- Two dev cases were re-run on the same code first, across all nine modes. Every
  pass/fail and every metric matched the committed dev results, so this run is
  comparable with them.
- No change between the dev run and this commit touches selection. The commits
  since then fix yt-dlp invocation, an error message, model loading and the web
  page.

## Result

| Condition | Held-out pass rate [95% CI] | vs full Gist (pp) [95% CI] | Dev pass rate (same 23-case dev split) |
| :-- | :-- | :-- | :-- |
| Full Gist | **58.3% (7/12)** [33.3, 83.3] | — | 82.6% (19/23) |
| Full Gist, heuristics off | 58.3% [33.3, 83.3] | +0.0 [+0.0, +0.0] | |
| Split budget, intent-aware | 66.7% [41.7, 91.7] | +8.3 [−16.7, +33.3] | 52.2% (12/23) |
| Transcript only | 58.3% [33.3, 83.3] | +0.0 [−33.3, +33.3] | 13.0% (3/23) |
| Uniform sampling | 16.7% [0.0, 41.7] | −41.7 [−66.7, −16.7] | 13.0% (3/23) |
| Visual only | 8.3% [0.0, 25.0] | −50.0 [−75.0, −25.0] | 65.2% (15/23) |
| Score top-k, no diversity | 0.0% [0.0, 0.0] | −58.3 [−83.3, −33.3] | |

Paired percentile bootstrap, 10,000 resamples (`bootstrap-held-out.md`). Per-case
output in `ablation-held-out.json`.

## Reading it

**Full Gist passes 7 of 12 unseen cases, against 19 of 23 on the dev split.**
The drop of about 24 points is large, and with 12 cases the interval is wide:
33% to 83%. The dev figure lies inside that interval, so the data cannot say how
much of the drop is overfitting to the dev cases and how much is these six
recordings being harder. It does say the dev figure should not be quoted as
Gist's expected performance on new video.

**What holds on unseen data.** Full Gist beats uniform sampling by 41.7 points,
visual-only by 50 and relevance-only top-k by 58.3, and none of those intervals
reaches zero. The coverage heuristics again change nothing (100% agreement), as
on the dev split.

**What does not.** On the dev split full Gist was far ahead of transcript-only
(82.6% against 13.0%); here they tie at 58.3%. The held-out split is 8 of 12
`speech_semantic` cases by construction, which is where transcript retrieval is
strongest.

## The failure mode it exposed

Five cases failed. Two are a pattern the dev split did not show:

| Case | Category | What happened |
| :-- | :-- | :-- |
| paul-graham-startup-killer | speech_semantic | Right moment (timestamp hit 1.0), but Gist kept a **frame** of the speakers, not the speech; with a budget of one item, no words survived. Transcript-only passed. |
| paul-graham-startup-ideas-unconsciously | speech_semantic | The same: right moment, frame instead of speech. Transcript-only passed. |
| paul-graham-yc-support | global_summary | Right moment, frame instead of speech; transcript-only also failed. |
| kinect-person-demo | mixed_av | Wrong moment in every mode. |
| kinect-post-interface-title | temporal_before_after | Wrong moment; temporal-ordering questions failed in every mode on the dev split too. |

On a talking-head recording, a CLIP frame of the people speaking can outscore
the speech window at the same moment, and when the budget is one item the
arbitration picks the picture. Gist found the moment and then kept the wrong
modality of it. This is the clearest target for future work, but it is not
changed here: tuning the arbitration now would be tuning on the held-out split,
and this split cannot be run a second time as an unseen test.

## Reproduce

```bash
python -m gist.eval.ablation --dataset data/eval/long-video-quality.jsonl \
  --output-root .gist/held-out-run --json results/held-out/ablation-held-out.json \
  $(python -c "import json; print(' '.join('--case-id '+i for i in json.load(open('data/eval/splits/held-out.json'))['held_out']['case_ids']))")
python -m gist.eval.bootstrap --ablation results/held-out/ablation-held-out.json \
  --baseline full_gist --split data/eval/splits/held-out.json --use held_out
```

`run.log` is the log of the execution reported here.
