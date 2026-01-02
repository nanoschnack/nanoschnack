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

Quick bootstrap (RunPod or fresh VM):

```sh
curl -fsSL https://raw.githubusercontent.com/nanoschnack/nanoschnack/refs/heads/main/scripts/bootstrap.sh | bash
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

To override batch size without editing code, set `BATCH_SIZE`:

```sh
BATCH_SIZE=16 python model/training.py
```

To adjust background data loading workers for streaming, set `DATA_LOADER_WORKERS`
(default: `0`):

```sh
DATA_LOADER_WORKERS=4 python model/training.py
```

Configure training datasets with `DATASET_SPECS` (comma-separated):

- `hf:<repo_id>[:split][:text_key]`
- `hf:<repo_id>:<config>:<split>[:text_key]`
- `txt:<path>[:text_key]`

Examples:

```sh
DATASET_SPECS="hf:coral-nlp/german-commons:web:onemillionposts:text,txt:data/goethe.txt:text" \
  python model/training.py
```

```sh
DATASET_SPECS="txt:data/goethe.txt:body" python model/training.py
```

## Dataset pipeline

1) parse_dataset_specs  
`dataset_specs = parse_dataset_specs(config.DATASET_SPECS)`

2) load_dataset_from_spec (streaming)  
`raw_dataset = load_dataset_from_spec(spec, cache_dir=data_dir, streaming=True)`

3) resolve_total_rows  
`total_rows = resolve_total_rows(raw_dataset, spec, cache_dir=data_dir)`

4) load checkpoint / normalize_resume_rows  
`resume_epoch, global_step, resume_samples, resume_state = checkpointer.load_latest()`  
`resume_rows = normalize_resume_rows(resume_state, dataset_specs)`

5) rank0 warm cache + dist.barrier  
`warm_dataset = load_dataset_from_spec(spec, cache_dir=data_dir, streaming=True)`  
`next(iter(warm_dataset.take(1)))`  
`dist.barrier()`

6) resolve_resume_plan  
`data_files, in_shard_offset, shard_label = resolve_resume_plan(spec, row_offset, cache_dir=data_dir)`

7) load_dataset_from_spec (data_files)  
`raw_streaming = load_dataset_from_spec(spec, cache_dir=data_dir, streaming=True, data_files=data_files)`

8) skip(in_shard_offset) + skip(row_offset)  
`raw_streaming = raw_streaming.skip(in_shard_offset)`  
`raw_streaming = raw_streaming.skip(row_offset)`

9) DDP split: filter(idx % world_size == rank)  
`raw_streaming = raw_streaming.filter(lambda _, idx: idx % ddp_world_size == ddp_rank, with_indices=True)`

10) tokenizer.map -> input_ids  
`tokenized = dataset.map(tokenizer_batch, batched=True, remove_columns=...)`

11) pack_tokens -> fixed blocks  
`packed = tokenized.map(lambda batch: pack_tokens(...), batched=True, batch_size=pack_batch_size)`

12) least-tokens interleave  
`base_dataset = build_interleaved_dataset(packed_datasets, seed=42, token_counts=resume_tokens)`

13) shuffle(buffer) -> DataLoader  
`dataset_epoch = base_dataset.shuffle(buffer_size=config.SHUFFLE_BUFFER, seed=...)`  
`loader = DataLoader(dataset_epoch, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.DATA_LOADER_WORKERS)`

14) row/token aggregation -> SUM across ranks  
`dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)`  
`resume_state = build_resume_state(global_counts, dataset_specs, source_token_counts=global_token_counts, seeded_specs=seeded_specs)`

## Configuration

Training and data settings live in `model/config.py`. See that file for the
full list of knobs and their defaults.  
`model/config.py`

To run distributed training with 8 GPUs:

```sh
torchrun --standalone --nproc_per_node=8 model/training.py
```

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
