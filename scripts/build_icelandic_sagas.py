#!/usr/bin/env python3
"""Materialize the Icelandic Saga Corpus as plain UTF-8 text.

Downloads a revision-pinned parquet artifact containing 49 saga texts.
Preserves the source text while producing input for the corpus builder.
Keeps the large generated text file outside version control.
"""

import argparse
from pathlib import Path

from datasets import load_dataset


SAGA_DATASET_REVISION = "f3b00d4f1d2983adc6748dbe3c4720951c2aba6c"
SAGA_PARQUET_URL = (
    "https://huggingface.co/datasets/danish-foundation-models/icelandic-dynaword/resolve/"
    f"{SAGA_DATASET_REVISION}/data/saga/saga.parquet"
)


def build_saga_corpus(output_path, cache_dir=None):
    # Load the revision-pinned Saga Corpus artifact.
    dataset = load_dataset(
        "parquet",
        data_files=SAGA_PARQUET_URL,
        split="train",
        cache_dir=cache_dir,
    )

    # Preserve paragraphs and separate documents with an extra newline.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            handle.write(text)
            handle.write("\n\n")
            document_count += 1
    print(f"Wrote {document_count} Icelandic saga documents to {output_path}", flush=True)
    return document_count


def main(argv=None):
    # Parse materialization options.
    parser = argparse.ArgumentParser(description="Materialize the Icelandic Saga Corpus.")
    parser.add_argument("--output", default="data/icelandic-sagas.txt", help="output text file path")
    parser.add_argument("--cache-dir", default=None, help="cache directory for the parquet artifact")
    args = parser.parse_args(argv)

    # Write the corpus for use by build_tokenizer_corpus.py.
    build_saga_corpus(args.output, cache_dir=args.cache_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
