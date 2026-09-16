# Split-budget vs pooled competition — does Gist's allocation choice matter?

Gist standardises both modalities against their own distributions (z-scores, per
modality) and then runs **one pooled competition** for the whole budget. Several
2026 omni-modal methods — OmniScope, Macer, OmniDelta — do the opposite: they
**allocate an explicit per-modality budget first**, then rank within each modality.

Until this run, the honest answer to "is yours better?" was *unmeasured*. The six
existing ablation modes vary the input signal and the post-processing; none varied
the allocation strategy. This run measures it.

## Result

39 curated cases, paired percentile bootstrap, 10,000 resamples, 95% intervals.

| Condition | Pass rate [95% CI] | vs full Gist (pp) | Agreement |
| :--- | :--- | :--- | ---: |
| **Full Gist (standardise, then pool)** | **69.2% [53.8%, 84.6%]** | — | — |
| Coverage heuristics OFF | 66.7% [51.3%, 82.1%] | −2.6 [−7.7, +0.0] | 97% |
| Split first, intent-aware | 51.3% [35.9%, 66.7%] | **−17.9 [−33.3, −2.6]** | 72% |
| Split first, even halves | 25.6% [12.8%, 41.0%] | **−43.6 [−59.0, −28.2]** | 56% |
| Visual-only retrieval | 41.0% [25.6%, 56.4%] | −28.2 [−43.6, −15.4] | 72% |
| Transcript-only retrieval | 25.6% [12.8%, 38.5%] | −43.6 [−61.5, −25.6] | 46% |
| Uniform sampling | 15.4% [5.1%, 28.2%] | −53.8 [−69.2, −35.9] | 41% |

**Pooling wins, and the interval excludes zero** for both split rules. Against the
stronger of the two (intent-aware), pooling is better by at least 2.6 points at 95%
confidence.

**The split rule matters enormously.** An even 50/50 split scores 25.6%; an
intent-aware split scores 51.3%. Testing only the naive rule would have overstated
the case by roughly 26 points, which is why two rules were run.

## The load-bearing caveat: the budget is almost always one item

| Budget full Gist settled on | Cases |
| :--- | ---: |
| 1 item | 32 |
| 2 items | 4 |
| 3 items | 1 |
| 4 items | 2 |

**82% of cases select a single item.** At a budget of one, "splitting the budget"
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
on all 39 cases**.

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
both fixed split rules on this corpus with intervals that exclude zero; and the
mechanism is understood (blind modality commitment at unit budgets).

**Cannot say:** that Gist's allocation beats OmniScope's, Macer's or OmniDelta's.
Those methods allocate with their own signals, on their own benchmarks, post-encoder.
This compares *allocation shapes* inside Gist, not published systems against
each other.
