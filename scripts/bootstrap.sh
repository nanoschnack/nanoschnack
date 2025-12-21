#!/usr/bin/env bash
set -euo pipefail

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
git clone https://github.com/nanoschnack/nanoschnack.git
cd nanoschnack
uv sync
source .venv/bin/activate
