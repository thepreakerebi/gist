# Bootstrap confidence intervals

Percentile bootstrap, 10,000 resamples, cases resampled together so the comparison stays paired. Intervals are 95%.

| Condition | Pass rate [95% CI] | vs baseline (pp) | Agreement |
| :--- | :--- | :--- | ---: |
| full_gist_no_heuristics | 66.7% [51.3%, 82.1%] | -2.6% [-7.7%, +0.0%] | 97% |
| split_intent | 51.3% [35.9%, 66.7%] | -17.9% [-33.3%, -2.6%] | 72% |
| split_intent_sep | 51.3% [35.9%, 66.7%] | -17.9% [-33.3%, -2.6%] | 72% |
| visual_only | 41.0% [25.6%, 56.4%] | -28.2% [-43.6%, -15.4%] | 72% |
| transcript_only | 25.6% [12.8%, 38.5%] | -43.6% [-61.5%, -25.6%] | 46% |
| score_topk | 25.6% [12.8%, 41.0%] | -43.6% [-59.0%, -28.2%] | 56% |
| split_even | 25.6% [12.8%, 41.0%] | -43.6% [-59.0%, -28.2%] | 56% |
| uniform | 15.4% [5.1%, 28.2%] | -53.8% [-69.2%, -35.9%] | 41% |

Baseline: **full_gist**, 69.2% [53.8%, 84.6%], n=39.

