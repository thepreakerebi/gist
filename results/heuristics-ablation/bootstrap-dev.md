# Component ablation — dev split (27 cases)

Percentile bootstrap, 10,000 resamples, cases resampled together so the comparison stays paired. Intervals are 95%.

| Condition | Pass rate [95% CI] | vs baseline (pp) | Agreement | Non-inferior |
| :--- | :--- | :--- | ---: | :--- |
| full_gist_no_heuristics | 70.4% [51.9%, 85.2%] | -3.7% [-11.1%, +0.0%] | 96% | no |
| visual_only | 55.6% [37.0%, 74.1%] | -18.5% [-33.3%, -3.7%] | 81% | no |
| score_topk | 37.0% [18.5%, 55.6%] | -37.0% [-55.6%, -18.5%] | 63% | no |
| uniform | 14.8% [3.7%, 29.6%] | -59.3% [-77.8%, -37.0%] | 33% | no |
| transcript_only | 11.1% [0.0%, 22.2%] | -63.0% [-81.5%, -44.4%] | 37% | no |

Baseline: **full_gist**, 74.1% [55.6%, 88.9%], n=27. Non-inferiority margin is one standard error of the baseline (±8.4%), as pre-registered.
