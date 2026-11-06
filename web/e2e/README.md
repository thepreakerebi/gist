# End-to-end tests

Playwright tests for the web app. The test cases, their expected results and the
latest outcome are tabulated in [`docs/testing/functional-tests.md`](../../docs/testing/functional-tests.md).

| Suite | File | Needs |
| :-- | :-- | :-- |
| offline | `offline.spec.ts` | the frontend only; the API is refused inside the browser |
| live | `live.spec.ts` | the API on port 8100 and a Postgres database with pgvector |

Both run in the installed Google Chrome (`channel: "chrome"`), because
Playwright's bundled Chromium has no H.264 decoder and the evidence clips are
H.264.

## Offline suite

```bash
cd web
bunx playwright install chromium   # first time only, for the test runner
bun run test:e2e
```

Playwright starts `next dev` on port 3100 itself.

## Live suite

The live suite writes to the database it is given, adds and removes a video, and
empties the library before it starts. Point it at a throwaway database, never at
the demo library.

```bash
# 1. A disposable Postgres with pgvector
docker run -d --name gist-e2e-pg -e POSTGRES_PASSWORD=gist -e POSTGRES_DB=gist \
  -p 55432:5432 pgvector/pgvector:pg16

# 2. The API against it, from an empty working directory so ingested media
#    lands there and not in the repository's .gist/library
mkdir -p /tmp/gist-e2e && cd /tmp/gist-e2e
DATABASE_URL=postgresql://postgres:gist@127.0.0.1:55432/gist \
TWELVELABS_API_KEY= \
GIST_CORS_ORIGINS=http://localhost:3100 \
  /path/to/gist/.venv/bin/uvicorn gist.api.app:app --host 127.0.0.1 --port 8100

# 3. Both suites
cd /path/to/gist/web
bun run test:e2e:live

# 4. Clean up
docker rm -f gist-e2e-pg
```

`TWELVELABS_API_KEY` is set empty on purpose. The suite rewrites each question's
`answerer` to `extractive` in flight, so it tests Gist's own selection and spends
no TwelveLabs credits; the empty key is a second guard against a hosted call.

The live suite issues itself a fresh, non-admin access code with
`python -m gist.access.cli create` against `GIST_E2E_DATABASE_URL` (default: the
container above), so its library starts empty and the daily cap of five applies.
Sign-in is limited to five attempts a minute per address, so wait a minute between
back-to-back runs. Both suites reach the API as `localhost`, not `127.0.0.1`: the
session cookie is only sent when the page and the API share a site.

The API is started by its absolute path, without activating the venv. That is how
a systemd unit starts it, and it is the condition under which ingestion used to
fail (see FT-11 in the functional test table).

## Output

`e2e-results/` (gitignored) holds the HTML report, the JSON results and, for every
test, a trace, a video and a final screenshot:

```bash
bunx playwright show-report e2e-results/html
bunx playwright show-trace e2e-results/artifacts/<test>/trace.zip
```

The final screenshot of each case is copied to `docs/testing/evidence/FT-xx.png`
as the committed record.
