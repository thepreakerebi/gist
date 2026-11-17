#!/usr/bin/env bash
# One-time setup of the Gist API on a fresh Ubuntu 24.04 server.
# Run as root:  bash setup.sh
# Safe to re-run: it updates the checkout and reinstalls.
set -euo pipefail

REPO=https://github.com/thepreakerebi/gist.git
APP=/srv/gist

apt-get update
apt-get install -y git ffmpeg curl debian-keyring debian-archive-keyring apt-transport-https ufw

# Caddy from its official repository.
if ! command -v caddy >/dev/null; then
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update && apt-get install -y caddy
fi

# Swap, so loading CLIP, CLAP and Whisper together never hits the OOM killer.
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

id gist >/dev/null 2>&1 || useradd --system --create-home --home-dir "$APP" --shell /bin/bash gist

if [ -d "$APP/.git" ]; then
  sudo -u gist git -C "$APP" pull --ff-only
else
  sudo -u gist git clone "$REPO" "$APP.tmp"
  shopt -s dotglob; mv "$APP.tmp"/* "$APP"/; rmdir "$APP.tmp"; chown -R gist:gist "$APP"
fi

# uv, then CPU-only torch. The lockfile pins CUDA wheels for Linux, which
# would add several GB of NVIDIA libraries this server cannot use.
sudo -u gist bash -c 'command -v ~/.local/bin/uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'
sudo -u gist bash -c "cd $APP && ~/.local/bin/uv venv --python 3.12 .venv && \
  ~/.local/bin/uv pip install --python .venv/bin/python --torch-backend=cpu -e '.[vision,audio,sound]'"

sudo -u gist mkdir -p "$APP/.gist/library"
[ -f "$APP/.env" ] || { cp "$APP/deploy/env.production.example" "$APP/.env"; chown gist:gist "$APP/.env"; chmod 600 "$APP/.env"; echo "Fill in $APP/.env"; }

cp "$APP/deploy/gist-api.service" /etc/systemd/system/gist-api.service
cp "$APP/deploy/Caddyfile" /etc/caddy/Caddyfile
systemctl daemon-reload
systemctl enable gist-api caddy
systemctl restart gist-api caddy

ufw allow OpenSSH && ufw allow 80/tcp && ufw allow 443/tcp && ufw --force enable
echo "Done. Check: systemctl status gist-api; journalctl -u gist-api -f"
