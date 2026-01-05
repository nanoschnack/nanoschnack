#!/usr/bin/env python3
"""Build a post-training QA corpus from Facebook MLQA German subset."""

import argparse
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import config
from tokenizer import load_tokenizer

ASSISTANT_TOKEN = "<|ASSISTANT|>"
END_TOKEN = "<|END|>"
USER_TOKEN = "<|USER|>"


def _normalize_text(text):
    # Collapse newlines to keep one conversation per line.
    if text is None:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\xa0", " ")
    return normalized.replace("\n", "\\n").rstrip()


def _format_message(role, text):
    # Normalize trailing whitespace so the end token always follows cleanly.
    cleaned = _normalize_text(text)
    if role == "user":
        return f"{USER_TOKEN}{cleaned}{END_TOKEN}"
    elif role == "assistant":
        return f"{ASSISTANT_TOKEN}{cleaned}{END_TOKEN}"
    raise ValueError(f"Unknown role: {role}")


def _span_token_count(tokenizer, spans):
    return len(tokenizer.encode("".join(spans)).ids)


MLQA_URL = "https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip"


def build_mlqa_de(output_path, limit, context_len, include_context):
    """Download MLQA German and convert to post-training format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()

    # Download and extract the dataset.
    print(f"Downloading {MLQA_URL}...", flush=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "mlqa.zip"
        urllib.request.urlretrieve(MLQA_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # Find the German test file (context-de-question-de).
        json_file = Path(tmpdir) / "MLQA_V1" / "test" / "test-context-de-question-de.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Expected file not found: {json_file}")

        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

    # Parse SQuAD format: {"data": [{"paragraphs": [{"context": ..., "qas": [...]}]}]}
    examples = []
    for article in data.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                question = qa.get("question", "")
                answers = qa.get("answers", [])
                if answers and question:
                    examples.append((context, question, answers[0].get("text", "")))

    print(f"Loaded {len(examples)} examples from MLQA German", flush=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for context, question, answer_text in examples:
            if limit is not None and written >= limit:
                break

            if not answer_text:
                continue

            # Build the user prompt.
            if include_context and context:
                user_text = f"Kontext: {context}\\n\\nFrage: {question}"
            else:
                user_text = question

            user_span = _format_message("user", user_text)
            assistant_span = _format_message("assistant", answer_text)

            # Check token count.
            if _span_token_count(tokenizer, [user_span, assistant_span]) > context_len:
                continue

            handle.write(f"{user_span}{assistant_span}\n")
            written += 1

    print(f"Wrote {written} QA pairs to {output_path}", flush=True)


def main(argv=None):
    # Parse CLI arguments.
    parser = argparse.ArgumentParser(
        description="Build the MLQA German post-training corpus."
    )
    parser.add_argument(
        "--output",
        default="data/posttraining/Facebook/MLQA-de.txt",
        help="output text file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional max number of QA pairs to export",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=config.CONTEXT_LEN,
        help="maximum token length for post-training spans",
    )
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="include the context passage in user prompts",
    )
    args = parser.parse_args(argv)

    build_mlqa_de(
        output_path=args.output,
        limit=args.limit,
        context_len=args.context_len,
        include_context=args.include_context,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
