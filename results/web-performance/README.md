# Web performance and network conditions

Page speed of the web app (Lighthouse), and the time each step of the main user
flow takes on slower connections. Tables are in [`summary.md`](summary.md),
generated from the committed JSON by `summarize.py`.

## Setup

| Item | Value |
| :-- | :-- |
| Frontend | production build (`next build`, `next start`) on port 3200 |
| Backend | the local stack in `web/e2e/README.md`: API on port 8100, Postgres with pgvector in Docker |
| Video | Video-MME `eZ5lFBPpRkM`, about an hour, already ingested |
| Browser | Google Chrome, headless |
| Lighthouse | 12.8.2, mobile and desktop presets |
| Machine | Apple M4 laptop; frontend, API and browser on the same machine |

Lighthouse ran on the library page and on a video page as a person first sees
it, with no questions asked yet. The video page was re-measured after the fix
below, on the same video ingested again (id changed, content identical). The mobile preset was run twice: with
Lighthouse's default simulated throttling, which loads the page at full speed
and estimates a slow phone, and with `--throttling-method=devtools`, which
applies the slow-4G network and 4x CPU slowdown and measures.

The network-condition runs (`web/scripts/network-scenarios.mjs`) drive the real
flow three times per profile under Chrome DevTools network emulation: open the
library, open the video, ask, wait for the evidence, open an evidence item and
wait until its clip can play. Questions use the extractive answerer.

## Findings

**Every step completes on every profile, including Slow 3G** (400 kbps, 400 ms
latency): 12 of 12 runs. On Slow 3G a page takes about 6 s to become usable and
a question about 2.4 s to show its evidence, while on Slow 4G both stay under
2 s. The answer path holds up better than page load because a question moves a
few kilobytes of events; the first page load moves about 690 KB of JavaScript.

**Accessibility, best practices and SEO score 100 on both pages, both form factors.**

**The library page meets the LCP target** once throttling is actually applied:
1.58 s median on slow 4G. The simulated 3.03 s is an estimate that the applied
run does not reproduce.

**The video page missed the LCP target on a slow phone, and was fixed.** Before:
3.20 s median against 2.5 s with throttling applied, performance 84 to 87 (raw
reports in `lighthouse/before-server-render/`). LCP equalled FCP: the page
rendered entirely in the browser, so nothing contentful painted until the
JavaScript had loaded and its own request to the API had returned. A small
loading label would not have helped, because LCP is the largest paint and the
real content arriving later would replace it.

The fix (`web/app/v/[id]/page.tsx`) fetches the video on the server when the API
answers within 1.5 s, so the first HTML already holds the page; on any failure
it hands over to the client exactly as before, which keeps the offline snapshot
working. After: 1.58 s median, performance 99 to 100. With the API stopped, the
page still returned in about 0.1 s and replayed the cached run, and all 17
functional tests pass.

**A video page with a conversation cannot be scored for LCP at all.** On load it
scrolls smoothly to the latest question, and Chrome stops recording LCP once a
page scrolls, so Lighthouse reports `NO_LCP` and no performance score. The page
itself paints in 0.2 s (desktop) to 0.8 s (mobile, simulated). The scores above
are for the page before any question, which is also the state a new viewer sees.

## Reproduce

```bash
# stack from web/e2e/README.md, then:
cd web
NEXT_PUBLIC_GIST_API=http://127.0.0.1:8100 bun run build && bunx next start -p 3200
# the API's GIST_CORS_ORIGINS must include http://localhost:3200

npx lighthouse@12.8.2 http://localhost:3200/v/<id> --throttling-method=devtools \
  --chrome-flags="--headless=new" --output=json --output-path=<file>
node scripts/network-scenarios.mjs <id> > ../results/web-performance/network.json
python ../results/web-performance/summarize.py
```

The JSON reports open in the Lighthouse Viewer
(https://googlechrome.github.io/lighthouse/viewer/).

## What this does not show

- **A real phone or a real mobile network.** Throttling is emulated on a laptop.
- **A deployed server.** Frontend, API and browser share one machine, so there
  is no real distance to the server beyond the emulated latency.
- **The hosted answer.** TwelveLabs adds its own time after the evidence is shown.
