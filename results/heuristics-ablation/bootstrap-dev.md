# Bootstrap confidence intervals

Percentile bootstrap, 10,000 resamples, cases resampled together so the comparison stays paired. Intervals are 95%.

| Condition | Pass rate [95% CI] | vs baseline (pp) | Agreement |
| :--- | :--- | :--- | ---: |
| full_gist_no_heuristics | 71.4% [57.1%, 85.7%] | -2.9% [-8.6%, +0.0%] | 97% |
| split_intent | 57.1% [40.0%, 74.3%] | -17.1% [-34.3%, +0.0%] | 71% |
| split_intent_sep | 57.1% [40.0%, 74.3%] | -17.1% [-34.3%, +0.0%] | 71% |
| visual_only | 45.7% [28.6%, 62.9%] | -28.6% [-42.9%, -14.3%] | 71% |
| transcript_only | 28.6% [14.3%, 42.9%] | -45.7% [-65.7%, -25.7%] | 43% |
| score_topk | 28.6% [14.3%, 42.9%] | -45.7% [-62.9%, -28.6%] | 54% |
| split_even | 28.6% [14.3%, 42.9%] | -45.7% [-62.9%, -28.6%] | 54% |
| uniform | 14.3% [2.9%, 25.7%] | -60.0% [-74.3%, -42.9%] | 40% |

Baseline: **full_gist**, 74.3% [60.0%, 88.6%], n=35.

