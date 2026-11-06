# Edge-case inputs

Videos and questions outside what the evaluation corpus covers, sent through the
real library path: the video is added by URL, downloaded by yt-dlp, ingested by
the scorers into Postgres, and questioned through the API. The question is
behaviour, not accuracy: does the system finish, fail with a message a person
can act on, or fail silently?

Run by `run_edge_cases.py` on 2026-11-06 at commit `da0661e`; raw output in
`edge_cases.json`. Synthetic videos are generated with ffmpeg (a moving test
pattern) and the macOS French voice "Thomas", and served over local HTTP.
Questions use the extractive answerer, so the evidence shown is Gist's own
selection and no hosted model is involved.

## Results

| Case | Input | Ingestion | Questions completed | What Gist selected | Verdict |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Silent audio track | 60 s pattern, silent audio | ready, 128 frames, 2 windows | 2 of 2 | for "what does the speaker say", the two silent audio windows | Works; returns silence as evidence rather than saying there is no speech |
| No audio stream | 60 s pattern, no audio track | ready, 0 audio windows | 1 of 1 | frames only | Works |
| Music, no speech | 60 s pattern, stepped melody | ready, 2 windows | 2 of 2 | both audio windows | Works; the 30 s windows are too coarse to place a change in a 60 s clip |
| French speech | 13 s pattern, French speech | ready, 1 window | 3 of 3 | the speech window, transcribed in French by Whisper, for English and French questions | Works; with one window, cross-lingual ranking is not tested |
| Three-second clip | 3 s pattern, tone | ready, 127 frames | 1 of 1 | frames at 2.60 s and 2.63 s | Works; 127 frames sampled from 75 real ones, so frames repeat |
| One still image, ten minutes | the same frame for 600 s | ready, 128 frames, 20 windows | 1 of 1 | three frames at 253 s, 272 s and 295 s, all the same picture | Works; redundant evidence |
| Question the video cannot answer | "What is the capital of Canada?" on the Bali video | — | 1 of 1 | four clips from across the video | **Gist never abstains** |
| Question in Chinese | 他们在哪里看日落？ on the Bali video | — | 1 of 1 | frames from early in the video | Completes; CLIP and MiniLM are English models, so the ranking is not meaningful |
| Question of only emoji | 🐢🐢🌅 | — | 1 of 1 | narration about Lembongan and Manta Point | Completes |
| Question of about 2,000 characters | "Where do they go snorkelling?" × 70 | — | 1 of 1 | **108 of 236 candidates unscored** | **Defect, fixed** (below) |
| Question of spaces only | five spaces | — | rejected, HTTP 422 | — | Works: "query must not be blank" (the web form also disables Ask) |

## Defect found and fixed

**A long question silently dropped every audio score.** CLAP's text encoder has
512 positions; the question tokenised to 636 and raised an index error.
`embed_query` caught it and returned no audio vector, so all 108 audio windows
were ranked on lexical overlap alone, and the person asking saw nothing wrong.
Fixed in commit `73aa1c9`: an over-long prompt is cut to fit before it reaches
the encoder, and a prompt within the limit passes through byte-identical, so no
existing score changes (`tests/test_clap_prompt_fit.py`; the stored-embedding
equivalence tests pass with real models). Re-run after the fix: 236 of 236
candidates scored. `edge_cases.json` keeps the original run as the record.

## Behaviour to know about, not fixed

- **No abstention.** Gist always returns its best evidence, even for a question
  the video cannot answer or for silence. Whether the person is told "this is
  not in the video" depends entirely on the answering model downstream. A
  relevance floor below which Gist reports no evidence would make this explicit.
- **Identical frames are not merged.** The diversity term in MMR compares
  timestamps, not picture content, so a static scene yields several copies of
  the same frame. On real footage this mostly matters for slides held on screen.
- **Short clips are oversampled.** The fixed 128-frame sample exceeds the real
  frame count below about 5 s at 25 fps, so some frames are scored twice.
- **English-only scorers.** CLIP and MiniLM are English models. Whisper
  transcribes French correctly, but ranking a long recording in another
  language, or a question in another language, is untested here and unlikely
  to be reliable.
- **The extractive answerer returns no answer text** on these inputs. The web app
  answers through TwelveLabs; the extractive path exists for tests and for
  running without a hosted model.

## Reproduce

Start the stack in `web/e2e/README.md`, ingest any long video, then:

```bash
python results/edge-cases/run_edge_cases.py --video <id>
```
