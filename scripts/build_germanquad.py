#!/usr/bin/env python3
"""Build a post-training QA corpus from GermanQuAD Kaggle dataset."""

import argparse
import ast
import csv
from pathlib import Path
import re
import sys

import kagglehub

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


def _parse_answers(answers_str):
    """Parse the answers field which contains text wrapped in array notation."""
    if not answers_str:
        return None

    # The field is formatted like: "{'text': array(['answer text'], dtype=object), ...}"
    # We extract the text between array([' and '], dtype).
    match = re.search(r"'text':\s*array\(\['([^']+)'", answers_str)
    if match:
        return match.group(1).strip()

    # Fallback: try ast.literal_eval for simple formats.
    try:
        parsed = ast.literal_eval(answers_str)
        if isinstance(parsed, dict) and "text" in parsed:
            texts = parsed["text"]
            if isinstance(texts, list) and texts:
                return texts[0]
            return texts
        return str(parsed)
    except (ValueError, SyntaxError):
        return answers_str


def build_germanquad(output_path, limit, context_len, include_context):
    """Download GermanQuAD and convert to post-training format."""
    # Download the dataset via kagglehub.
    dataset_path = kagglehub.dataset_download(
        "thedevastator/germanquad-high-quality-german-qa-dataset"
    )
    dataset_path = Path(dataset_path)
    print(f"Dataset downloaded to: {dataset_path}", flush=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer()

    # Find CSV files in the dataset.
    csv_files = list(dataset_path.glob("*.csv"))
    if not csv_files:
        csv_files = list(dataset_path.glob("**/*.csv"))
    print(f"Found CSV files: {csv_files}", flush=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for csv_file in csv_files:
            if "test" in csv_file.name.lower():
                continue
            print(f"Processing: {csv_file}", flush=True)
            with csv_file.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                # Print available columns for debugging.
                print(f"Columns: {reader.fieldnames}", flush=True)
                for row in reader:
                    if limit is not None and written >= limit:
                        break

                    # Extract fields - try common QA dataset column names.
                    question = row.get("question") or row.get("Question") or ""
                    context = row.get("context") or row.get("Context") or ""
                    answers = row.get("answers") or row.get("Answers") or row.get("answer") or ""

                    # Parse the answer.
                    answer_text = _parse_answers(answers)
                    if not answer_text or not question:
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

            if limit is not None and written >= limit:
                break

    print(f"Wrote {written} QA pairs to {output_path}", flush=True)


def main(argv=None):
    # Parse CLI arguments.
    parser = argparse.ArgumentParser(
        description="Build the GermanQuAD post-training corpus from Kaggle."
    )
    parser.add_argument(
        "--output",
        default="data/posttraining/Kaggle/GermanQuAD.txt",
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

    build_germanquad(
        output_path=args.output,
        limit=args.limit,
        context_len=args.context_len,
        include_context=args.include_context,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
