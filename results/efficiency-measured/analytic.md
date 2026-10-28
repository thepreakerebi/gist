# Gist system-efficiency — analytic (architecture-derived FLOPs, real item counts)

- Encoder profile: qwen2.5-omni-7b; baseline: full baseline
- Frames/windows are real encode counts. GFLOPs are **derived** from the encoder
  configs, not profiled — do not quote them as measured.

| Condition | Frames | Audio windows | System GFLOPs | Downstream tokens | GFLOPs saving |
| :--- | ---: | ---: | ---: | ---: | ---: |
| full baseline | 64 | 120 | 313,345 | 102,544 | 0.0% |
| uniform-K | 8 | 4 | 14,351 | 4,568 | 95.4% |
| gist-K | 2.778 | 3.167 | 8,995 | 2,920 | 97.1% |
