# Deployment

The live demo runs as two parts on one registrable domain, as the session
cookie requires (see `docs/security.md`):

| Part | Host | Address |
|---|---|---|
| Frontend (`web/`) | Vercel | `https://gist.usesky.io` |
| API (`gist.api.app`) | Ubuntu 24.04 server, Caddy in front | `https://api.gist.usesky.io` |

The answering model is TwelveLabs Pegasus, so the server holds only CLIP, CLAP
and faster-whisper on CPU: about 4 GB of memory, 8 GB recommended. Qwen2.5-Omni
is not deployed.

## Files

- `setup.sh`: one-time server setup, run as root. Installs ffmpeg and Caddy,
  adds 4 GB of swap, clones the repository to `/srv/gist`, installs CPU-only
  torch with the `vision`, `audio` and `sound` extras, and enables the services.
- `gist-api.service`: systemd unit. One uvicorn worker on `127.0.0.1:8000` with
  `--proxy-headers`, working directory `/srv/gist`, so the library is
  `/srv/gist/.gist/library` and the API reads `/srv/gist/.env`.
- `Caddyfile`: TLS from Let's Encrypt, security headers, unbuffered proxying
  for progress streams and byte-range video.
- `env.production.example`: the server's `.env`.

## Steps

1. DNS: an `A` record `api.gist` pointing to the server, and the `gist`
   record Vercel gives when the domain is added to the project.
2. On the server: `bash setup.sh`, then fill in `/srv/gist/.env` and
   `systemctl restart gist-api`.
3. Vercel: project root `web`, environment variables
   `NEXT_PUBLIC_GIST_API=https://api.gist.usesky.io` and
   `NEXT_PUBLIC_API_BASE=https://api.gist.usesky.io`, domain `gist.usesky.io`.
4. Warm the models with one question; the first load takes about two minutes.
5. Issue codes on the server: `sudo -u gist /srv/gist/.venv/bin/gist-access ...`.

## Known limits

- YouTube often refuses downloads from data-centre addresses. Videos already
  in the library keep working, because ingestion happens once.
- Whisper runs on CPU, so a one-hour video takes 10 to 20 minutes to ingest.
- Updating: `bash setup.sh` again pulls and reinstalls.
