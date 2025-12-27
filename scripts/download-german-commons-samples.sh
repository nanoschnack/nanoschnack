#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-.}"
target_bytes=$((1024 * 1024))

python - "$out_dir" "$target_bytes" <<'PY'
import os
import sys

try:
    from datasets import get_dataset_split_names, load_dataset
    from datasets.utils.logging import set_verbosity_error
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: datasets. Install with `pip install datasets`."
    ) from exc

out_dir = sys.argv[1]
target_bytes = int(sys.argv[2])
os.makedirs(out_dir, exist_ok=True)
set_verbosity_error()

subsets = ["web", "cultural", "news"]


def extract_text(row):
    if "text" not in row:
        raise SystemExit("Expected a 'text' column but none was found in the dataset.")
    value = row["text"]
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return "\n".join(value)
    return None


def write_sample(subset, split):
    out_path = os.path.join(out_dir, f"samples-{subset}-{split}.txt")
    if os.path.exists(out_path):
        print(f"{subset}/{split}: {out_path} exists, skipping")
        return
    dataset = load_dataset(
        "coral-nlp/german-commons",
        name=subset,
        split=split,
        streaming=True,
    )
    written = 0
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in dataset:
            text = extract_text(row)
            if not text:
                continue
            payload = text.rstrip("\n") + "\n---\n"
            encoded = payload.encode("utf-8")
            if written + len(encoded) > target_bytes:
                remaining = target_bytes - written
                chunk = encoded[:remaining]
                handle.write(chunk.decode("utf-8", errors="ignore"))
                written += len(chunk)
                break
            handle.write(payload)
            written += len(encoded)
            if written >= target_bytes:
                break
    print(f"{subset}/{split}: wrote {written} bytes to {out_path}")


for subset in subsets:
    try:
        splits = get_dataset_split_names("coral-nlp/german-commons", subset)
    except Exception:
        dataset = load_dataset("coral-nlp/german-commons", name=subset)
        splits = list(dataset.keys())
    for split in splits:
        write_sample(subset, split)
PY
