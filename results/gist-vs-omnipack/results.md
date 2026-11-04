# Gist against the OmniPack rebuild, same input, model, prompt and GPU

Device: NVIDIA GeForce RTX 4090, torch 2.6.0+cu124, CUDA 12.4.

Questions: 48 over 17 videos. Idle board power: 27.06 W.

Audio heard by every arm: 300 to 300 s of videos lasting 32 to 58 min; the 32 frames span the whole video.

## Accuracy

| Arm | Correct / answered | 95% CI | Media tokens to the LLM (median) | Share of full | Gist minus arm, paired [95% CI] |
| :-- | --: | :-- | --: | --: | :-- |
| full | 22/48 | [31.2%, 60.4%] | 9,804 | 100.0% | -6.2% [-16.7%, +2.1%] (n=48) |
| omnipack_r0.25 | 22/48 | [31.2%, 60.4%] | 2,451 | 25.0% | -6.2% [-14.6%, +0.0%] (n=48) |
| omnipack_r0.2 | 22/48 | [31.2%, 60.4%] | 1,961 | 20.0% | -6.2% [-14.6%, +0.0%] (n=48) |
| omnipack_r0.15 | 23/48 | [33.3%, 62.5%] | 1,471 | 15.0% | -8.3% [-16.7%, -2.1%] (n=48) |
| omnipack_r0.1 | 20/48 | [27.1%, 56.2%] | 980 | 10.0% | -2.1% [-8.3%, +4.2%] (n=48) |
| gist | 19/48 | [25.0%, 54.2%] | 1,776 | 18.1% |  |
| uniform | 20/48 | [27.1%, 56.2%] | 1,476 | 15.1% | -2.1% [-10.4%, +6.2%] (n=48) |

## Cost on the same GPU

Per video is paid once for every question about that video; per question is paid each time.
Decoding the video is shared by every arm and not included: median 4.4 s per video.

| Method | Per video: GPU s | J | TFLOPs | Per question: GPU s | J | TFLOPs | Max working memory GB |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| full | 1.83 | 536 | 37.51 | 1.24 | 542 | 169.95 | 1.969 |
| omnipack_r0.25 | 2.47 | 605 | 38.32 | 0.41 | 139 | 30.51 | 1.047 |
| omnipack_r0.2 | 2.48 | 607 | 38.33 | 0.35 | 119 | 24.55 | 1.061 |
| omnipack_r0.15 | 2.49 | 612 | 38.35 | 0.29 | 98 | 18.75 | 1.075 |
| omnipack_r0.1 | 2.51 | 611 | 38.36 | 0.24 | 77 | 13.07 | 1.09 |
| gist | 5.42 | 493 | 0.97 | 0.59 | 191 | 33.01 | 0.731 |
| uniform | 0.00 | 0 | 0.00 | 0.50 | 166 | 29.16 | 0.671 |

## When Gist costs less in total

Total cost for n questions about one video is per-video cost + n x per-question cost.

| Against | GPU time | Energy | FLOPs |
| :-- | :-- | :-- | :-- |
| full | beyond 5.5 questions | always | always |
| omnipack_r0.25 | never | up to 2.1 questions | up to 15.0 questions |
| omnipack_r0.2 | never | up to 1.6 questions | up to 4.4 questions |
| omnipack_r0.15 | never | up to 1.3 questions | up to 2.6 questions |
| omnipack_r0.1 | never | up to 1.0 questions | up to 1.9 questions |

## Whisper setting (sensitivity)

Selection used Whisper tiny (beam 1) on speech segments only: mean 4.51 s and 399 J per video. The product's setting, base (beam 3) on every segment: mean 5.52 s and 520 J per video.

## Encoding only the chosen units

Relative error of the visual tokens against an fp32 reference (exactness.txt):

```
EXACT ZLRyTFIZOIs visual max_abs 0.9609 rel_fro 1.37e-02 min_cos 0.969626 max_abs_value 24.47
EXACT ZLRyTFIZOIs audio max_abs 0.0 rel_fro 0.00e+00 min_cos 1.0 max_abs_value 9.2
EXACT eZ5lFBPpRkM visual max_abs 2.2075 rel_fro 2.38e-02 min_cos 0.859806 max_abs_value 25.12
EXACT eZ5lFBPpRkM audio max_abs 0.0 rel_fro 0.00e+00 min_cos 1.0 max_abs_value 15.19
FP32REF ZLRyTFIZOIs fp32 4v1 1.49e-05 | fp16 groups-of-4 vs fp32 1.73e-02 | fp16 one-at-a-time vs fp32 1.70e-02
FP32REF eZ5lFBPpRkM fp32 4v1 1.66e-05 | fp16 groups-of-4 vs fp32 2.32e-02 | fp16 one-at-a-time vs fp32 2.76e-02
```

In fp32, encoding one temporal patch at a time equals encoding four at a time (1e-5); in fp16 both paths sit about 2% from fp32. Audio is identical.

FLOPs: PyTorch stages counted with `FlopCounterMode` on the counting pass and carried to every video and question by fitted per-unit rates; Whisper's are derived from its architecture.
