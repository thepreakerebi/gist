# Load test: concurrent users against the library API

How the API behaves as more people use it at once, and where it stops meeting its
budgets. Results are in [`load.md`](load.md); the k6 summary for every level is in
`raw/`.

## Setup

| Item | Value |
| :-- | :-- |
| Machine | Apple M4, 10 cores, 16 GB, macOS 27.2; laptop CPU, no GPU |
| API | `uvicorn gist.api.app:app`, one process, as the README starts it |
| Database | PostgreSQL 16.15 with pgvector, in Docker on the same machine |
| Scorers | CLIP, CLAP, MiniLM and faster-whisper at the commits in `src/gist/model_revisions.py` |
| Video | Video-MME `eZ5lFBPpRkM` (item 868), about an hour long, ingested once before the runs |
| Load generator | k6 v2.3.0, on the same machine |
| Answerer | extractive (`answerer: "extractive"`), so no hosted model is called |

Each level holds a constant number of virtual users for 60 s. Every user repeats
one action, then waits 1 s:

- **ask**: post one of the three Video-MME questions about the video and read the
  whole event stream (scoring, selection, clip cutting with ffmpeg, answer).
- **varied**: as ask, but drawing from 40 different questions, with the video's
  cut-clip cache emptied before each level, so most requests select new evidence
  and cut new clips with ffmpeg. This is the closer model of many people asking
  their own questions.
- **browse**: list the library, then open the video.

The conversation is cleared between levels so its size does not carry over.

## Budgets

Fixed in `analyze.py` and committed (478b8fb) before the first run: question
stream P95 ≤ 5 s, library reads P95 ≤ 300 ms, at most 1% of iterations failing.

## Results

All levels finished with 0% errors; every question stream ended with a `done`
event. Full tables in [`load.md`](load.md).

| Scenario | Within budget up to | Breaking point | At the breaking point |
| :-- | :-- | :-- | :-- |
| Browse (list, open) | 200 users, P95 ≤ 0.09 s | not reached | — |
| Three repeated questions | 100 users, P95 2.36 s | not reached | — |
| 40 different questions, cold clip cache | 25 users, P95 1.20 s | 50 users | P95 5.51 s, P99 17.6 s; throughput falls from 22.8 to 18.9 questions/s at 100 users |

**Where the time goes.** Scoring a question against stored embeddings takes
milliseconds; the pgvector query itself took 7 ms. The cost that grows with load
is cutting evidence clips with ffmpeg, which a new selection needs and a repeated
one finds on disk. That is why the varied scenario breaks first, and why its tail
(P99 11.3 s at 25 users) rises well before its P95 does. Running ffmpeg cuts in a
bounded worker pool, or cutting clips only when the viewer opens one, are the
next things to change if the server must carry more than about 25 simultaneous
askers.

### Defect found: the encoders were reloaded for every question

The first run (rows under "Before the encoder fix" in `load.md`, raw data in
`raw/before-fix/`) missed the budget with a single user: P95 5.66 s, rising to
59.3 s at 50 users. Timing one question's stream put all of it in "Scoring
stored evidence", and the API log showed CLIP and CLAP loading their weights on
every request. `gist.library.query` built a new scorer per question, so each
question reloaded both models' weights from disk. Fixed in commit `df563f2`:
one instance per encoder per process, created behind a lock, with inference
serialised per encoder so a shared tokenizer cannot fail silently under
concurrency (`tests/test_library_query_scorers.py`).

This also qualifies an earlier figure. The 1.26 s median for cached selection in
`efficiency-measured/` was measured by `cached_selection.py`, which keeps its
scorers loaded. The web app's query path did not, so before this fix a question
in the web app cost several seconds, not about one.

## Reproduce

Start the stack in `web/e2e/README.md` (Postgres in Docker, API on port 8100),
ingest the video, then:

```bash
VIDEO=<id> ./results/load-test/run.sh ask    "1 5 10 25 50 100"
VIDEO=<id> ./results/load-test/run.sh browse "1 10 50 100 200"
VIDEO=<id> CLIP_DIR=<library>/<id>/clips ./results/load-test/run.sh varied "1 5 10 25 50 100"
python results/load-test/analyze.py
```

## What this does not show

- **A deployed server.** The API, database and load generator share one laptop,
  so k6 competes with the API for CPU. The demo server planned for the defense is
  a different machine.
- **The hosted answer.** TwelveLabs latency and its rate limits are outside Gist
  and are not exercised.
- **Ingestion under load.** Ingestion runs once per video in a background thread
  and takes minutes; it is not part of the per-question path measured here.
- **Many different videos.** Every question targets one ingested video.
