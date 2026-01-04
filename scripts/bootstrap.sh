#!/usr/bin/env bash
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v go >/dev/null 2>&1; then
  curl -LsSf https://go.dev/dl/go1.25.0.linux-amd64.tar.gz -o /tmp/go1.25.0.linux-amd64.tar.gz
  rm -rf "$HOME/.local/go" && tar -C "$HOME/.local" -xzf /tmp/go1.25.0.linux-amd64.tar.gz
  export PATH="$HOME/.local/go/bin:$PATH"
fi
git clone https://github.com/nanoschnack/nanoschnack.git
cd nanoschnack
uv sync
source .venv/bin/activate
