# nanoschnack

## Goals

- Create a nanochat inspired chat bot (https://nanochat.ai).
- Train on German internet data only.
- Success criteria: you can chat in German on http://nanoschnack.de.

## Development setup

Install uv:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install dev tooling (including pre-commit and jupytext):

```sh
uv sync --extra dev
```

Activate the uv-managed virtual environment:

```sh
source .venv/bin/activate
```

Install the pre-commit hook:

```sh
pre-commit install
```

## Data download

The training data is sourced from the `arnomatic/german-wikipedia-clean-no-lists` dataset:
https://huggingface.co/datasets/arnomatic/german-wikipedia-clean-no-lists

Training downloads parquet shards on demand into `data/` and caches them locally.

## Training

Open the training notebook and run the cells:

```sh
jupyter lab model/training.ipynb
```

You can also run the generated script (kept in sync with the notebook):

```sh
python model/training.py
```

Training loads one shard at a time, shuffles within the shard, and writes checkpoints to `checkpoints/`.
Resuming training uses the stored shard position from the latest checkpoint.

To override batch size without editing code, set `NANOSCHNACK_BATCH_SIZE`:

```sh
NANOSCHNACK_BATCH_SIZE=16 python model/training.py
```

For multi-GPU training with `torchrun`, set a per-process batch size and launch one process per GPU:

```sh
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1
NANOSCHNACK_BATCH_SIZE=32 torchrun --standalone --nproc_per_node=8 model/training.py
```

Only rank 0 prints logs and writes checkpoints; other ranks stay quiet.

## Inference (chat)

Run the REPL chat interface:

```sh
python model/chat.py
```

Options:
- `--checkpoint /path/to/checkpoint.pt` (defaults to `checkpoints/latest.pt` if present)
- `--context-len 256`
- `--max-new-tokens 128`
- `--temperature 0.8`
- `--top-k 50`
