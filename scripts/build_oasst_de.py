#!/usr/bin/env python3
"""Build a post-training chat corpus from OpenAssistant OASST-DE."""

import argparse
from pathlib import Path
import sys

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import config
from tokenizer import load_tokenizer

ASSISTANT_TOKEN = "<|ASSISTANT|>"
END_TOKEN = "<|END|>"
SYSTEM_TOKEN = "<|SYSTEM|>"
USER_TOKEN = "<|USER|>"

ROLE_TOKENS = {
    "assistant": ASSISTANT_TOKEN,
    "prompter": USER_TOKEN,
    "user": USER_TOKEN,
    "system": SYSTEM_TOKEN,
}


def _normalize_text(text):
    # Collapse newlines to keep one conversation per line.
    if text is None:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return normalized.replace("\n", "\\n").rstrip()


def _format_message(role, text):
    # Normalize trailing whitespace so the end token always follows cleanly.
    cleaned = _normalize_text(text)
    token = ROLE_TOKENS.get(role)
    if token is None:
        raise ValueError(f"Unknown role: {role}")
    return f"{token}{cleaned}{END_TOKEN}"


def _span_token_count(tokenizer, spans):
    return len(tokenizer.encode("".join(spans)).ids)


def build_oasst_de(output_path, split, limit, context_len):
    # Load the dataset locally and write to the post-training format.
    dataset = load_dataset("OpenAssistant/OASST-DE", split=split)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(dataset):
            if limit is not None and idx >= limit:
                break

            conversation = row.get("conversation") or []
            if not conversation:
                continue

            spans = []
            system_text = None
            pair_index = 0
            i = 0
            while i < len(conversation):
                message = conversation[i]
                role = message.get("role")
                text = message.get("text", "")
                if role == "system" and not spans:
                    system_text = text
                    i += 1
                    continue
                if role not in ("prompter", "user"):
                    i += 1
                    continue
                if i + 1 >= len(conversation):
                    break
                next_message = conversation[i + 1]
                if next_message.get("role") != "assistant":
                    break
                user_span = _format_message("prompter", text)
                assistant_span = _format_message("assistant", next_message.get("text", ""))

                # Enforce context limits on the first system/user/assistant trio, then on user/assistant pairs.
                if pair_index == 0 and system_text:
                    system_span = _format_message("system", system_text)
                    if _span_token_count(tokenizer, [system_span, user_span, assistant_span]) > context_len:
                        spans = []
                        break
                    spans.append(system_span)
                if _span_token_count(tokenizer, [user_span, assistant_span]) > context_len:
                    break
                spans.extend([user_span, assistant_span])
                pair_index += 1
                i += 2

            if not spans:
                continue
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
    parser.add_argument(
        "--context-len",
        type=int,
        default=config.CONTEXT_LEN,
        help="maximum token length for post-training spans",
    )
    args = parser.parse_args(argv)

    # Build the dataset file.
    build_oasst_de(
        output_path=args.output,
        split=args.split,
        limit=args.limit,
        context_len=args.context_len,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
