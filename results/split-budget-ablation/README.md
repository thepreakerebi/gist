# Split-budget vs pooled competition — does Gist's allocation choice matter?

Gist standardises both modalities against their own distributions (z-scores, per
modality) and then runs **one pooled competition** for the whole budget. Several
2026 omni-modal methods — OmniScope, Macer, OmniDelta — do the opposite: they
**allocate an explicit per-modality budget first**, then rank within each modality.

Until this run, the honest answer to "is yours better?" was *unmeasured*. The six
existing ablation modes vary the input signal and the post-processing; none varied
the allocation strategy. This run measures it.

## Result

> **Re-run 2026-09-22 at n=35, and one conclusion changed.** The original run was
> n=39. Four cases were withdrawn because they ran against
> `tears_of_steel_61min.webm`, which was the 12.2-minute Blender film concatenated
> five times rather than an hour-long recording. The result against the
> *intent-aware* split no longer excludes zero. That claim is withdrawn below; it
> is not softened in place, because the n=39 version of this file asserted the
> opposite.

35 curated cases, paired percentile bootstrap, 10,000 resamples, 95% intervals.

| Condition | Pass rate [95% CI] | vs full Gist (pp) | Agreement |
| :--- | :--- | :--- | ---: |
| **Full Gist (standardise, then pool)** | **74.3% [60.0%, 88.6%]** | — | — |
| Coverage heuristics OFF | 71.4% [57.1%, 85.7%] | −2.9 [−8.6, +0.0] | 97% |
| Split first, intent-aware | 57.1% [40.0%, 74.3%] | −17.1 [−34.3, **+0.0**] | 71% |
| Split first, even halves | 28.6% [14.3%, 42.9%] | **−45.7 [−62.9, −28.6]** | 54% |
| Visual-only retrieval | 45.7% [28.6%, 62.9%] | −28.6 [−42.9, −14.3] | 71% |
| Transcript-only retrieval | 28.6% [14.3%, 42.9%] | −45.7 [−65.7, −25.7] | 43% |
| Uniform sampling | 14.3% [2.9%, 25.7%] | −60.0 [−74.3, −42.9] | 40% |

**Pooling beats the naive even split decisively**, by at least 28.6 points at 95%
confidence, and that interval is nowhere near zero.

**Against the intent-aware split, pooling no longer wins at 95% confidence.** The
point estimate still favours pooling by 17.1 points, but the interval runs to
+0.0: across 10,000 resamples the split rule was never better, yet the data at
n=35 cannot exclude parity. At n=39 the same comparison gave [−33.3, −2.6] and
this file claimed the interval excluded zero. **That claim is withdrawn.** Four
cases were enough to move it, which is the honest reading of a 35-case corpus, not
a defect in the resampling.

What survives: the allocation choice was tested rather than asserted, pooling is
never worse, and the naive rule is clearly worse. What does not survive: any
statement that pooling is *significantly* better than a well-chosen split rule.

**The split rule matters enormously.** An even 50/50 split scores 28.6%; an
intent-aware split scores 57.1%. Testing only the naive rule would have overstated
the case by roughly 29 points, which is why two rules were run.

## The load-bearing caveat: the budget is almost always one item

| Budget full Gist settled on | Cases |
| :--- | ---: |
| 1 item | 29 |
| 2 items | 4 |
| 4 items | 2 |

**83% of cases select a single item.** At a budget of one, "splitting the budget"
degenerates into *committing to one modality before looking at the evidence*. That
is the worst possible regime for split-then-rank and the best possible one for
pooling, and it explains the mechanism directly: an even split at budget 1 always
takes the visual slot, which is why `split_even` averages 0.13 audio items per case
against full Gist's 0.51, and why it collapses on speech and sound-event questions.

So the defensible claim is **not** "splitting is worse in general". It is:

> On this corpus, where the adaptive budget almost always settles on a single
> item, allocating that budget by modality in advance costs 18 to 44 points
> depending on the rule, because it forces a blind modality commitment that
> pooling defers until the scores are known.

At larger budgets a split has room to hedge, and the gap would plausibly narrow.
This run does not measure that regime.

## Joint scoring buys nothing once the selection is split

`split_intent` and `split_intent_sep` — the latter scoring each modality in complete
isolation rather than sharing one scored pool — produced **byte-identical selections
on all 35 cases**.

That is correct, not a wiring bug. z-scores are computed per modality in
`_score_modality` either way, and `_apply_audio_visual_anchors` only *annotates*
visual candidates; it changes selection solely through `_is_audio_visual_anchor_pair`
inside cross-modal redundancy, which never fires when MMR runs within a single
modality. **Gist's cross-modal machinery only pays off inside a pooled competition.**

## How the comparison is held fair

Everything is identical except the competition structure: same candidate pool, same
scoring, same z-scores, same anchors, same MMR, same coverage heuristics, same answer
pipeline, same preset config that `full_gist` settled on, and a total budget taken
from what `full_gist` actually selected for that case.

The split modes score the full pool once, partition the scored candidates by
modality, and run Gist's own `_select_with_mmr` within each partition at its
allocated share. Neither split rule is trained — both are fixed policies, which is
the point: a training-free system cannot learn an allocation policy, so the honest
comparison is against fixed rules.

No production code was modified. Both modes live in `src/gist/eval/ablation.py`.

## Reproduce

```bash
# the candidate cache is per output root; seed it or this takes hours
cp -R .gist/ablation-heuristics-ab/cache .gist/ablation-split2/cache

python -m gist.eval.ablation \
  --dataset data/eval/long-video-quality.jsonl \
  --output-root .gist/ablation-split2 \
  --json results/split-budget-ablation/split-budget-ablation.json \
  --markdown results/split-budget-ablation/split-budget-ablation.md

python -m gist.eval.bootstrap \
  --ablation results/split-budget-ablation/split-budget-ablation.json \
  --baseline full_gist
```

## What this does and does not license saying

**Can say:** the allocation choice was tested rather than asserted; pooling beats
the even split on this corpus with an interval that excludes zero by a wide
margin; pooling is never worse than the intent-aware split across 10,000
resamples; and the mechanism is understood (blind modality commitment at unit
budgets).

**Cannot say (changed 2026-09-22):** that pooling significantly beats the
intent-aware split. At n=35 that interval reaches +0.0.

**Cannot say:** that Gist's allocation beats OmniScope's, Macer's or OmniDelta's.
Those methods allocate with their own signals, on their own benchmarks, post-encoder.
This compares *allocation shapes* inside Gist, not published systems against
each other.
