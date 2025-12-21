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

The training data is sourced from the `pdelobelle/fineweb-german-edu-mt` dataset:
https://huggingface.co/datasets/pdelobelle/fineweb-german-edu-mt

Training downloads shards automatically into `data/` as needed.

## Training

Open the training notebook and run the cells:

```sh
jupyter lab model/training.ipynb
```

The notebook uses streaming parquet loading from `data/*.parquet` and writes checkpoints to `checkpoints/`.

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
