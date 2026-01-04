#!/usr/bin/env python3
"""Print packed post-training blocks as sent to the trainer."""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = REPO_ROOT / "model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

import config
import loader
from tokenizer import load_tokenizer


def print_blocks(path, block_size, limit, show_text):
    # Force post-training mode for loss masks and filtering.
    config.POST_TRAINING = True
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing post-training file: {path}")
    dataset = load_dataset("text", data_files=str(path), split="train")
    tokenizer = load_tokenizer()
    docs_total = getattr(dataset, "num_rows", None)
    user_id = tokenizer.token_to_id(loader.USER_TOKEN)
    assistant_id = tokenizer.token_to_id(loader.ASSISTANT_TOKEN)
    packed = loader.build_packed_dataset(
        dataset,
        tokenizer=tokenizer,
        block_size=block_size,
        text_key="text",
        pack_batch_size=1000,
        source_id=0,
    )

    printed = 0
    user_spans = 0
    assistant_spans = 0
    input_count = 0
    loss_token_count = 0
    for row in packed:
        input_ids = row["input_ids"]
        loss_mask = row.get("loss_mask")
        input_list = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        mask_list = (
            loss_mask.tolist()
            if loss_mask is not None and hasattr(loss_mask, "tolist")
            else (list(loss_mask) if loss_mask is not None else None)
        )
        print(f"Block {printed}: tokens={len(input_list)}")
        print(f"  input_ids={input_list}")
        if mask_list is not None:
            print(f"  loss_mask={mask_list}")
            print(f"  loss_tokens={sum(mask_list)}")
        if show_text:
            print("  text:")
            decoded = tokenizer.decode(input_list, skip_special_tokens=False)
            for line in decoded.splitlines() or [""]:
                print(f"    {line}")
        printed += 1
        input_count += len(input_list)
        if user_id is not None:
            user_spans += sum(1 for token in input_list if token == user_id)
        if assistant_id is not None:
            assistant_spans += sum(1 for token in input_list if token == assistant_id)
        if mask_list is not None:
            loss_token_count += sum(mask_list)
            assistant_spans += sum(
                1 for idx, value in enumerate(mask_list)
                if value and (idx == 0 or not mask_list[idx - 1])
            )
        if limit and printed >= limit:
            break
    docs_label = docs_total if docs_total is not None else "unknown"
    print(
        "Stats: "
        f"docs={docs_label} blocks_printed={printed} "
        f"tokens_printed={input_count} loss_tokens={loss_token_count} "
        f"user_spans={user_spans} assistant_spans={assistant_spans}",
        flush=True,
    )


def main(argv=None):
    # Parse CLI arguments.
    parser = argparse.ArgumentParser(
        description="Print packed post-training blocks from a text dataset."
    )
    parser.add_argument(
        "--path",
        default="data/posttraining/OpenAssistant/OASST-DE.txt",
        help="path to the post-training text file",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=config.CONTEXT_LEN,
        help="context length for packing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="number of blocks to print",
    )
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="skip decoded text output",
    )
    args = parser.parse_args(argv)

    print_blocks(
        path=args.path,
        block_size=args.block_size,
        limit=args.limit,
        show_text=not args.no_text,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
