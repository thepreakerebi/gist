# Access and security

The hosted library is reachable only with an access code. This page describes
how access works, what each limit is for, and what a deployment must configure.

## Access codes

There are no accounts, emails or passwords. The operator issues a code and
hands it over directly:

```bash
uv run gist-access create --label "Me" --admin   # admin: sees every video, no daily cap
uv run gist-access create --label "Marvin" --expires-days 30
uv run gist-access list
uv run gist-access revoke <id>                   # ends its sessions at once; videos are kept
uv run gist-access replace <id>                  # lost code: new code, same videos, old one revoked
uv run gist-access transfer <old id> <new id>    # move videos between two existing codes
```

A code looks like `GIST-7K3M-QX9P-R2TA-H8WD`: 16 characters from Crockford's
base32 alphabet (80 bits of randomness). Input is forgiving about case, spaces,
dashes and the letters O, I and L. The first admin code created also takes
ownership of any videos added before access codes existed.

Only SHA-256 digests of codes and session tokens are stored
(`src/gist/access/codes.py`), so a copy of the database does not let anyone
sign in, and a lost code cannot be recovered, only replaced. `replace` does that
in one transaction: a new code with the same label, role and expiry date, every
video moved to it, and the old code revoked with its sessions ended. The new
code starts the day with a fresh question count.

## Sessions

Signing in (`POST /v1/auth/login`) exchanges a code for a session token in a
cookie:

| Property | Value | Why |
| :-- | :-- | :-- |
| `HttpOnly` | yes | page scripts cannot read the token |
| `SameSite` | `Lax` | the browser does not send it on requests started by other sites |
| `Secure` | when served over HTTPS | never sent in clear text in deployment |
| Lifetime | 1 day | then the code is asked for again |

A cookie rather than an `Authorization` header, because source video, clips
and frames load through `<video src>` and `<img src>`, which cannot attach
headers.

**Log out** (`POST /v1/auth/logout`) ends every session of the code, on every
device. The code stays valid: signing in with it again restores the same
library and the same count of questions used today. Revoking or expiring a code
ends its sessions immediately.

## Who can see what

Each video belongs to the code that added it (`videos.owner_code_id`). Every
library route, including the media files, checks ownership; another code's video
answers 404, the same as a video that does not exist, so ids cannot be probed.
Admin codes see and manage every video. A link is unique per owner, so two codes
adding the same video each get their own copy.

## Limits

| What | Limit | Keyed by | Purpose |
| :-- | :-- | :-- | :-- |
| Sign-in attempts | 5 a minute, 20 an hour | client address | code guessing |
| Questions | 5 a day (resets 00:00 Kigali time) | code | TwelveLabs cost per answer |
| Questions | 10 a minute | code | bursts |
| Adding videos | 3 an hour | code | transcription occupies the CPU for minutes |
| Ingestions running at once | 2 | whole server | the same, across all codes |
| Any other request | 600 a minute | code | runaway clients; generous for video byte ranges |

Admin codes are exempt from the per-code question and ingestion limits. The
daily count is stored in Postgres (`question_usage`), apart from the chat
history, so clearing a chat, deleting a video or signing out does not reset it.
A question that fails before producing an answer is given back. The per-minute
and per-hour limits are held in memory (`src/gist/access/ratelimit.py`), which is
exact for the single uvicorn process the API runs as and resets on restart.

## Other protections

- **Research endpoints are off by default.** `/v1/ingestions`,
  `/v1/local-video-compressions`, `/v1/compressions` and `/v1/demo/run` take
  paths on the server's disk and run the full pipeline; they are mounted only
  with `GIST_ENABLE_RESEARCH_ROUTES=1`, for local experiments.
- **Video ids must be UUIDs.** Anything else, including `..`, is rejected before
  it reaches a filesystem path. Clip and frame names must be bare filenames.
- **Cross-site requests.** CORS allows only the listed frontend origins, with
  credentials, and a wildcard is refused at startup (with credentials, Starlette
  would echo any origin). A `POST` or `DELETE` whose `Origin` is not listed is
  refused with 403 before authentication.
- **Input size.** Questions are capped at 1,000 characters and links at 2,048.
- **Sign-in redirect.** `/login?next=` accepts only same-site paths.

## Deployment requirements

1. **Frontend and API on one site.** Serve them from the same registrable
   domain, e.g. `gist.example.com` and `api.gist.example.com`. On different
   sites (a `*.vercel.app` frontend with an API elsewhere) the session cookie is
   a third-party cookie, which Safari blocks and `SameSite=Lax` never sends on
   `fetch`. Set `GIST_COOKIE_DOMAIN=gist.example.com` so the frontend's server
   also receives the cookie and can render the video page on the server.
2. **`GIST_CORS_ORIGINS`** set to the frontend's exact origin.
3. **Trust the proxy's client address.** Behind Caddy, start uvicorn with
   `--proxy-headers --forwarded-allow-ips=127.0.0.1`. Without it every request
   appears to come from the proxy, and all visitors share one sign-in limit.
4. **HTTPS** at the proxy, so the cookie carries `Secure`
   (`GIST_COOKIE_SECURE=1` forces it).
5. **Security headers** at the proxy, for example in the Caddyfile:

   ```
   header {
       Strict-Transport-Security "max-age=31536000"
       X-Content-Type-Options nosniff
       Referrer-Policy strict-origin-when-cross-origin
       X-Frame-Options DENY
   }
   ```

## Tests

- `tests/test_access.py`: code format and normalisation, the rate limiter, and
  the HTTP rules (401 without a session, 404 for another code's video, UUID ids,
  research routes unmounted, foreign origins refused, the daily cap, the cookie's
  attributes, logout ending every session). No database needed.
- `tests/test_access_store.py`: the SQL against a real Postgres, including the
  migration of a database from before access codes. Runs only with
  `GIST_TEST_DATABASE_URL` set to a throwaway database, because it drops tables.
- `web/e2e/live.spec.ts` FT-18 to FT-22: sign-in redirect, wrong code, sign-in,
  the sixth question refused, and Log out signing out everywhere.

The load test (`results/load-test/`) and network scenarios
(`web/scripts/network-scenarios.mjs`) were run before access codes existed and
call the API without a session; rerunning them needs a session cookie added.
