# Bootstrap confidence intervals, held-out split

Percentile bootstrap, 10,000 resamples, cases resampled together so the comparison stays paired. Intervals are 95%.

| Condition | Pass rate [95% CI] | vs baseline (pp) | Agreement |
| :--- | :--- | :--- | ---: |
| split_intent | 66.7% [41.7%, 91.7%] | +8.3% [-16.7%, +33.3%] | 75% |
| split_intent_sep | 66.7% [41.7%, 91.7%] | +8.3% [-16.7%, +33.3%] | 75% |
| full_gist_no_heuristics | 58.3% [33.3%, 83.3%] | +0.0% [+0.0%, +0.0%] | 100% |
| transcript_only | 58.3% [33.3%, 83.3%] | +0.0% [-33.3%, +33.3%] | 67% |
| uniform | 16.7% [0.0%, 41.7%] | -41.7% [-66.7%, -16.7%] | 58% |
| visual_only | 8.3% [0.0%, 25.0%] | -50.0% [-75.0%, -25.0%] | 50% |
| score_topk | 0.0% [0.0%, 0.0%] | -58.3% [-83.3%, -33.3%] | 42% |
| split_even | 0.0% [0.0%, 0.0%] | -58.3% [-83.3%, -33.3%] | 42% |

Baseline: **full_gist**, 58.3% [33.3%, 83.3%], n=12.
