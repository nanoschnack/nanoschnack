#!/usr/bin/env python3
"""Build a mixed German text corpus with equal character contributions per dataset."""

import argparse
import random
import sys
import time
from pathlib import Path

# Ensure model modules import cleanly (loader expects a top-level config module).
REPO_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import config
from loader import dataset_label, load_dataset_from_spec, parse_dataset_specs


def _normalize_text(value):
    # Collapse whitespace and drop empty rows.
    text = " ".join(str(value).split())
    return text if text else None


def _iter_equal_char_texts(specs, cache_dir, seed):
    # Stream datasets by always sampling from the least-emitted source.
    rng = random.Random(seed)
    datasets = []
    text_keys = []
    labels = []
    for spec in specs:
        datasets.append(load_dataset_from_spec(spec, cache_dir=cache_dir, streaming=True))
        text_keys.append(spec.get("text_key", "text"))
        labels.append(dataset_label(spec))

    iterators = [iter(dataset) for dataset in datasets]
    char_counts = [0] * len(iterators)
    active = list(range(len(iterators)))
    while active:
        min_count = min(char_counts[idx] for idx in active)
        candidates = [idx for idx in active if char_counts[idx] == min_count]
        choice = rng.choice(candidates) if len(candidates) > 1 else candidates[0]
        try:
            row = next(iterators[choice])
        except StopIteration:
            active.remove(choice)
            continue

        text = _normalize_text(row.get(text_keys[choice]))
        if text is None:
            continue

        char_len = len(text)
        char_counts[choice] += char_len
        yield text, choice, char_len, labels


def build_corpus(
    output_path,
    target_size,
    specs_str,
    cache_dir,
    seed,
    log_every,
):
    # Prepare inputs and output directory.
    specs = parse_dataset_specs(specs_str)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream text until the byte target is reached.
    bytes_written = 0
    row_counts = [0] * len(specs)
    char_counts = [0] * len(specs)
    next_log = log_every
    start = time.time()
    with output_path.open("w", encoding="utf-8") as handle:
        for text, idx, char_len, labels in _iter_equal_char_texts(
            specs,
            cache_dir,
            seed,
        ):
            line = f"{text}\n"
            payload = line.encode("utf-8")
            if bytes_written + len(payload) > target_size:
                break

            handle.write(line)
            bytes_written += len(payload)
            row_counts[idx] += 1
            char_counts[idx] += char_len
            if log_every and bytes_written >= next_log:
                elapsed = time.time() - start
                print(
                    f"Wrote {bytes_written} bytes in {elapsed:.1f}s "
                    f"({bytes_written / max(elapsed, 1e-6):.0f} B/s)",
                    flush=True,
                )
                next_log += log_every

    # Emit a final per-dataset summary.
    elapsed = time.time() - start
    print(
        f"Done: {bytes_written} bytes in {elapsed:.1f}s "
        f"({bytes_written / max(elapsed, 1e-6):.0f} B/s)",
        flush=True,
    )
    labels = [dataset_label(spec) for spec in specs]
    for idx, label in enumerate(labels):
        print(
            f"  {label}: rows={row_counts[idx]} chars={char_counts[idx]}",
            flush=True,
        )


def main(argv=None):
    # Parse CLI arguments.
    parser = argparse.ArgumentParser(
        description="Build a German tokenizer corpus with equal tokens per dataset."
    )
    parser.add_argument(
        "--output",
        default="data/tokenizer_corpus.txt",
        help="output text file path",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=1_000_000_000,
        help="target size in UTF-8 bytes",
    )
    parser.add_argument(
        "--specs",
        default=config.DATASET_SPECS,
        help="comma-separated dataset specs",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="cache directory for HF datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for tie-breaking",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100_000_000,
        help="log progress every N bytes (0 disables)",
    )
    args = parser.parse_args(argv)

    # Build the corpus with equal-token interleaving.
    build_corpus(
        output_path=args.output,
        target_size=args.size,
        specs_str=args.specs,
        cache_dir=args.cache_dir,
        seed=args.seed,
        log_every=args.log_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
