#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  exit 0
fi

if ! command -v jupytext >/dev/null 2>&1; then
  echo "jupytext not found in PATH" >&2
  exit 1
fi

files_to_add=()

for notebook in "$@"; do
  jupytext --sync "$notebook"
  files_to_add+=("$notebook")

  base="${notebook%.ipynb}"
  if [[ -f "${base}.py" ]]; then
    files_to_add+=("${base}.py")
  fi
  if [[ -f "${base}.md" ]]; then
    files_to_add+=("${base}.md")
  fi
done

git add "${files_to_add[@]}"
