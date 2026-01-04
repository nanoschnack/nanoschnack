#!/usr/bin/env python3
"""Build a post-training chat corpus from OpenAssistant OASST-DE."""

import argparse
from pathlib import Path

from datasets import load_dataset

ASSISTANT_TOKEN = "<|ASSISTANT|>"
END_TOKEN = "<|END|>"
SYSTEM_TOKEN = "<|SYSTEM|>"
USER_TOKEN = "<|USER|>"

ROLE_TOKENS = {
    "assistant": ASSISTANT_TOKEN,
    "prompter": USER_TOKEN,
    "system": SYSTEM_TOKEN,
}


def _format_message(role, text):
    # Normalize trailing whitespace so the end token always follows cleanly.
    cleaned = (text or "").rstrip()
    token = ROLE_TOKENS.get(role)
    if token is None:
        raise ValueError(f"Unknown role: {role}")
    return f"{token}{cleaned}{END_TOKEN}"


def build_oasst_de(output_path, split, limit):
    # Load the dataset locally and write to the post-training format.
    dataset = load_dataset("OpenAssistant/OASST-DE", split=split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            if limit is not None and idx >= limit:
                break

            conversation = row.get("conversation") or []
            if not conversation:
                continue

            spans = []
            for message in conversation:
                role = message.get("role")
                text = message.get("text", "")
                spans.append(_format_message(role, text))
            handle.write("".join(spans) + "\n")
            written += 1

    print(f"Wrote {written} conversations to {output_path}", flush=True)


def main(argv=None):
    # Parse CLI arguments.
    parser = argparse.ArgumentParser(
        description="Build the OpenAssistant OASST-DE post-training corpus."
    )
    parser.add_argument(
        "--output",
        default="data/posttraining/OpenAssistant/OASST-DE.txt",
        help="output text file path",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="dataset split to use",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional max number of conversations to export",
    )
    args = parser.parse_args(argv)

    # Build the dataset file.
    build_oasst_de(
        output_path=args.output,
        split=args.split,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
