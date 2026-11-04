# Gist against the OmniPack rebuild, same input, model, prompt and GPU

Device: NVIDIA GeForce RTX 4090, torch 2.6.0+cu124, CUDA 12.4.

Questions: 48 over 17 videos. Idle board power: 18.43 W.

Out of memory in the run of record, on a 24 GB card with Gist's scorers (1.3 GB) also loaded: omnipack_r0.25 13 (rerun without Gist's scorers on the card: 11 answered, 2 still out of memory); omnipack_r0.1 4 (rerun without Gist's scorers on the card: 4 answered, 0 still out of memory).

Audio heard by every arm: 1932 to 3506 s of videos lasting 32 to 58 min; the 128 frames span the whole video.

Not run because the prefill would exceed the model's 32,768-token context: full (48 question-arms; 57,642 to 97,057 tokens).

## Accuracy

| Arm | Correct / answered | 95% CI | Media tokens to the LLM (median) | Share of full | Gist minus arm, paired [95% CI] |
| :-- | --: | :-- | --: | --: | :-- |
| omnipack_r0.25 | 22/46 | [32.6%, 63.0%] | 18,954 |  | -4.3% [-17.4%, +8.7%] (n=46) |
| omnipack_r0.1 | 21/48 | [29.2%, 58.3%] | 7,698 |  | -2.1% [-18.8%, +12.5%] (n=48) |
| omnipack_r0.04 | 20/48 | [27.1%, 56.2%] | 3,033 |  | +0.0% [-14.6%, +14.6%] (n=48) |
| gist | 20/48 | [27.1%, 56.2%] | 2,088 |  |  |
| uniform | 19/48 | [25.0%, 54.2%] | 1,634 |  | +2.1% [-10.4%, +14.6%] (n=48) |

## Cost on the same GPU

Per video is paid once for every question about that video; per question is paid each time.
Decoding the video is shared by every arm and not included: median 13.7 s per video.

| Method | Per video: GPU s | J | TFLOPs | Per question: GPU s | J | TFLOPs | Max working memory GB |
| :-- | --: | --: | --: | --: | --: | --: | --: |
| omnipack_r0.25 | 22.25 | 4,528 | 267.78 | 4.65 | 1,515 | 274.04 | 4.55 |
| omnipack_r0.1 | 22.34 | 4,538 | 268.21 | 1.51 | 500 | 101.46 | 1.713 |
| omnipack_r0.04 | 22.51 | 4,553 | 268.39 | 0.60 | 191 | 38.22 | 1.608 |
| gist | 70.55 | 5,005 | 7.03 | 0.81 | 207 | 36.80 | 0.601 |
| uniform | 0.00 | 0 | 0.00 | 0.74 | 196 | 32.29 | 0.597 |

## When Gist costs less in total

Total cost for n questions about one video is per-video cost + n x per-question cost.

| Against | GPU time | Energy | FLOPs |
| :-- | :-- | :-- | :-- |
| omnipack_r0.25 | beyond 12.6 questions | always | always |
| omnipack_r0.1 | beyond 69.2 questions | beyond 1.6 questions | always |
| omnipack_r0.04 | never | never | always |

Exactness: encoding only the chosen units matched the full encoding to max |diff| 1.17 (visual) and 0 (audio).

FLOPs: PyTorch stages counted with `FlopCounterMode` on the counting pass and carried to every video and question by fitted per-unit rates; Whisper's are derived from its architecture.
