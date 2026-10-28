# Gist system-efficiency — measured (wall clock and peak GPU memory)

- Device: NVIDIA GeForce RTX 4090 25 GB, torch 2.6.0+cu124, CUDA 12.4
- Baseline: full; stage: answer
- First 1 successful call(s) per condition dropped as warmup.
- `activation GB` is peak allocated minus memory already resident, so the
  model weights (identical under every condition) do not mask the effect.
- `peak reserved GB` is the allocator high-water mark, which is what decides
  whether the card OOMs.

| Condition | n | OOM | Median s | Speedup | Activation GB (median) | Peak reserved GB | Activation saving |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 18 | 0 | 1.20 | 1.00x | 1.722 | 21.87 | 0.0% |
| gist | 18 | 1 | 0.88 | 1.37x | 1.216 | 21.34 | 29.4% |

**Out of memory.** gist exhausted the device on at least
one case. For a pre-encoder claim that is a result, not a run failure: the
condition did not fit on hardware where Gist did.
