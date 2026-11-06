# Functional test cases

End-to-end tests of the web app against its user stories. Each case is an
automated Playwright test (`web/e2e/`), run in Google Chrome at 1440 × 900. The
test drives the interface as a person would and does not inspect application
code. Every run records a trace, a video and a final screenshot per case; the
final screenshot is committed under [`evidence/`](evidence/).

How to run both suites: [`web/e2e/README.md`](../../web/e2e/README.md).

**Latest run:** 2026-11-06, 17 of 17 passed in 77 s, after the two defects below were fixed.

## Environments

| Suite | Backend | Answering model |
| :-- | :-- | :-- |
| offline | none; every API request refused in the browser, so the app runs on its cached-run snapshot | none; answers come from the snapshot of real TwelveLabs runs |
| live | FastAPI on port 8100, Postgres 16 with pgvector in Docker, real CLIP, CLAP and Whisper on a laptop CPU | extractive answerer, rewritten in flight, so no hosted model is called |

## User stories

| ID | Story |
| :-- | :-- |
| US-1 | As a viewer, I add a YouTube video once, see its progress, and find it in my library when it is ready. |
| US-2 | As a viewer, I ask a question about a video and get an answer with the evidence it came from. |
| US-3 | As a viewer, I check why each piece of evidence was kept and watch the clip itself. |
| US-4 | As a viewer, I come back later and find my questions still there, or clear them. |
| US-5 | As a viewer, I remove a video I no longer want. |
| US-6 | As a viewer, when something goes wrong, I am told what happened in words I can act on. |
| US-7 | As a presenter, the demonstration still works when the API or network is down. |

## Cases

| ID | Story | Steps | Expected result | Suite | Result | Evidence |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| FT-01 | US-7 | Open the library with the API unreachable | Snapshot videos listed, with an "Offline" notice saying the API is not reachable | offline | Pass | [FT-01](evidence/FT-01.png) |
| FT-02 | US-7 | From the offline library, open "Me at the zoo" | Video page opens at `/v/<id>` with the question box | offline | Pass | [FT-02](evidence/FT-02.png) |
| FT-03 | US-2, US-7 | Ask a question that has a recorded run | Answer and "Kept n of m candidates" shown, labelled "replayed from a cached run" | offline | Pass | [FT-03](evidence/FT-03.png) |
| FT-04 | US-3 | Ask, then expand the first evidence item | Item expands, shows "Why it was kept", and its clip loads in the player | offline | Pass | [FT-04](evidence/FT-04.png) |
| FT-05 | US-7 | Ask a question with no recorded run | Real evidence from another recorded run is shown, labelled as cached, not presented as live | offline | Pass | [FT-05](evidence/FT-05.png) |
| FT-06 | US-6 | Add a link while the API is down | Alert: "Can't reach the Gist API", with the command that starts it | offline | Pass | [FT-06](evidence/FT-06.png) |
| FT-07 | US-1 | Try to add with no link, with a link but no consent, with consent but only spaces | Add stays disabled until there is a non-blank link and consent is ticked | offline | Pass | [FT-07](evidence/FT-07.png) |
| FT-08 | US-2 | Leave the question empty, then type only spaces | Ask stays disabled | offline | Pass | [FT-08](evidence/FT-08.png) |
| FT-09 | US-6 | API rejects a link with a message (simulated 422) | The API's message is shown to the user | offline | Pass | [FT-09](evidence/FT-09.png) |
| FT-10 | US-6 | Library contains a video whose ingestion failed (simulated) | The failure reason is shown with a Retry button | offline | Pass | [FT-10](evidence/FT-10.png) |
| FT-11 | US-1 | Add a real 19-second video and wait | A progress bar appears, then the video becomes a ready link in the library | live | Pass after fix 1 | [FT-11](evidence/FT-11.png) |
| FT-12 | US-1 | Add the same link again | "Already in your library." and still only one entry | live | Pass | [FT-12](evidence/FT-12.png) |
| FT-13 | US-2, US-3 | Ask a question on the ingested video, expand the first evidence item | Live result (not labelled cached) with "Kept n of m candidates"; the item's reason shows and its clip loads from the API | live | Pass | [FT-13](evidence/FT-13.png) |
| FT-14 | US-4 | Reload the video page | The question asked in FT-13 is still shown | live | Pass | [FT-14](evidence/FT-14.png) |
| FT-15 | US-4 | Press Clear once, then again; reload | First press asks "Clear?" and keeps the history; second removes it; it stays gone after reload | live | Pass | [FT-15](evidence/FT-15.png) |
| FT-16 | US-6 | Add the text "not a link" | An alert explains what is wrong with the link, not only the HTTP status | live | Pass after fix 2 | [FT-16](evidence/FT-16.png) |
| FT-17 | US-5 | Remove the video; reload | The video disappears and the library is empty after reload | live | Pass | [FT-17](evidence/FT-17.png) |

## Defects found and fixed

| # | Found by | Observed | Cause | Fix |
| :-- | :-- | :-- | :-- | :-- |
| 1 | FT-11 | Every ingestion failed with "yt-dlp is not installed" | The API ran `yt-dlp` as an executable on PATH. Started by its absolute venv path, as a systemd unit starts it, the venv's `bin` is not on PATH, although the package is installed | yt-dlp now runs as `python -m yt_dlp` through the API's own interpreter (`src/gist/library/media.py`, commit `107f538`) |
| 2 | FT-16 | Adding "not a link" showed only "422 Unprocessable Entity" | FastAPI returns validation errors as a list of `{loc, msg}`; the library client read `detail` only when it was a string | `errorDetail` in `web/lib/library.ts` reads both forms; the user now sees "provide a full http(s) video URL" |

Before these fixes the latest run was 15 of 17: FT-11 failed (defect 1), and once it was fixed FT-16 failed (defect 2).

## What these cases do not cover

- **The hosted answer.** The live suite uses the extractive answerer, so the
  TwelveLabs step that writes the final answer in the deployed app is not run
  automatically. It is exercised by hand in the [video demo](https://youtu.be/Q_6_BG6ncAk),
  and the offline cases replay recorded TwelveLabs answers.
- **Long videos.** FT-11 ingests a 19-second video so the suite finishes in about a
  minute. Ingestion of 36- and 49-minute recordings was tested by hand
  (README, Testing).
- **Answer quality.** Whether the selected evidence answers the question is
  measured by the benchmark runs in `results/`, not here.
- **Browsers other than Chrome, and mobile widths.**
