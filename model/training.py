# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # NanoSchnack Model
#
# ## Setup
#
# - Install dependencies.
# - Verify that MPS is available (for Apple Silicon GPUs).

# %%
import torch
from device import device_info, pick_device, print_device_info

device = pick_device()
info = device_info(device)
print_device_info(info)

# Switch to TF32 for 8x speedup on supported hardware, and good enough for LLM training.
torch.set_float32_matmul_precision("high")

# Switch to TF32 for 8x speedup on supported hardware, and good enough for LLM training.
torch.set_float32_matmul_precision("high")


# %% [markdown]
# ## Loading a tokenizer with Hugging Face's tokenizer library
#
# - Compare: https://github.com/huggingface/tokenizers
# - Tiktokenizer: https://tiktokenizer.vercel.app/?model=gpt2

# %%
from tokenizer import load_tokenizer

tokenizer = load_tokenizer()

# %% [markdown]
# ## Instantiating the NanoSchnack model

# %%
from gpt import GPT
from autotune import find_max_batch_size
import config

# Resolve model paths so relative data/checkpoint locations are stable.
try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths
model_dir, data_dir, checkpoint_dir = setup_paths()

# Pull model sizes from the most recent checkpoint if present.
import torch
checkpoint_path = checkpoint_dir / "latest.pt"
if checkpoint_path.exists():
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_state, dict) and "config" in checkpoint_state:
        ckpt_config = checkpoint_state["config"]
        config.CONTEXT_LEN = ckpt_config.get("CONTEXT_LEN", config.CONTEXT_LEN)
        config.EMBED_SIZE = ckpt_config.get("EMBED_SIZE", config.EMBED_SIZE)
        config.NUM_LAYERS = ckpt_config.get("NUM_LAYERS", config.NUM_LAYERS)
        config.NUM_HEADS = ckpt_config.get("NUM_HEADS", config.NUM_HEADS)
        config.HIDDEN_SIZE = ckpt_config.get("HIDDEN_SIZE", config.HIDDEN_SIZE)

context_len = config.CONTEXT_LEN
embed_size = config.EMBED_SIZE
num_layers = config.NUM_LAYERS
num_heads = config.NUM_HEADS
hidden_size = config.HIDDEN_SIZE

# add special tokens
tokenizer.add_special_tokens(["[PAD]"])
pad_id = tokenizer.token_to_id("[PAD]")

model = GPT(
    vocab_size=tokenizer.get_vocab_size(),
    embed_size=embed_size,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_size=hidden_size,
    context_len=context_len,
).to(device).train()

# %% [markdown]
# ## Load the Training Data

# %%
from datasets.utils.logging import enable_progress_bar, set_verbosity_warning
from loader import (
    TokenEstimator,
    build_interleaved_dataset,
    build_packed_dataset,
    load_dataset_source,
)

# Download shards on demand and shuffle within each dataset.
set_verbosity_warning()
enable_progress_bar()

token_estimator = TokenEstimator(tokenizer)

# Build packed datasets per source.
dataset_specs = [
    {"repo_id": "arnomatic/german-wikipedia-clean-no-lists"},
    {"repo_id": "PatrickHaller/fineweb-2-de-1B"},
]
packed_datasets = []
for spec in dataset_specs:
    raw_streaming = load_dataset_source(
        spec["repo_id"],
        cache_dir=data_dir,
        streaming=True,
    )
    packed = build_packed_dataset(
        raw_streaming,
        tokenizer=tokenizer,
        block_size=config.CONTEXT_LEN,
        pack_batch_size=config.PACK_BATCH_SIZE,
    )
    packed_datasets.append(packed)

base_dataset = build_interleaved_dataset(packed_datasets, seed=42)
print(f"Packed dataset ready ({len(packed_datasets)} sources).", flush=True)



# %% [markdown]
# ## Run the Training

# %%
from plot import ascii_loss_plot
from progress import ProgressLogger
from checkpointer import Checkpointer
from scheduler import build_warmup_cosine
from torch.utils.data import DataLoader
import math
import os
import time

# Set up optimizer, learning-rate scheduler, and loss function
epochs = 1 # epochs between 1 and 3 are usually sufficient for good results, rather 1 than 3.
estimated_total_tokens = 0

print("Estimating tokens from dataset samples...", flush=True)
for dataset_index, spec in enumerate(dataset_specs):
    raw_dataset = load_dataset_source(
        spec["repo_id"],
        cache_dir=data_dir,
        streaming=True,
    )
    total_rows = None
    if raw_dataset.info and raw_dataset.info.splits:
        split_info = raw_dataset.info.splits.get("train")
        if split_info is not None:
            total_rows = split_info.num_examples
    if total_rows is None:
        raise ValueError("Dataset split metadata missing num_examples for token estimate.")
    avg_tokens, est_total_tokens = token_estimator.estimate_streaming(raw_dataset, total_rows)
    estimated_total_tokens += est_total_tokens
    print(
        f"Dataset {dataset_index + 1}/{len(dataset_specs)} "
        f"({spec['repo_id']}): avg_tokens={avg_tokens:.1f}, "
        f"est_tokens={est_total_tokens}"
    )
# Derive the token cap and epoch count from the configured max-training factor.
max_tokens = 0
if config.MAX_TRAINING_FACTOR > 0:
    max_tokens = int(param_count * config.MAX_TRAINING_FACTOR)
target_tokens = max_tokens if max_tokens else estimated_total_tokens
if max_tokens and estimated_total_tokens > 0:
    epochs = max(1, math.ceil(target_tokens / estimated_total_tokens))
tokens_per_sample = config.CONTEXT_LEN - 1
tokens_per_step = config.BATCH_SIZE * tokens_per_sample
steps_per_epoch = math.ceil(estimated_total_tokens / tokens_per_step)
total_steps = math.ceil(target_tokens / tokens_per_step)
print(f"Estimated steps per epoch: {steps_per_epoch} (total {total_steps}).", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
scheduler = build_warmup_cosine(optimizer, total_steps, config.WARMUP_PCT)
lossFn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

# The checkpointer will save and load model/optimizer/scheduler states to/from disk.
checkpointer = Checkpointer(checkpoint_dir, model, optimizer, scheduler, device=device)
resume_epoch, resume_step, global_step, sample_index, resume_tokens = checkpointer.load_latest()

last_ckpt_time = time.time()

# Initialize the progress logger to display training progress and loss
progress = ProgressLogger(
    ascii_loss_plot,
    start_global_step=global_step,
    start_total_samples=sample_index,
    start_total_tokens=resume_tokens,
    log_interval=config.LOG_INTERVAL_SECS,
    warmup_plot_interval=config.PLOT_WARMUP_SECS,
    plot_interval=config.PLOT_INTERVAL_SECS,
    warmup_window_secs=config.WARMUP_WINDOW_SECS,
    estimated_total_tokens=target_tokens,
)

last_epoch = resume_epoch
last_step = resume_step
# Resume from the saved sample index.
start_position = sample_index
current_position = start_position

# Enable debug output with DEBUG levels.
debug_level = int(os.getenv("DEBUG", "0"))
printed_debug_sample = False
try:
    print("Starting training loop...", flush=True)
    for epoch in range(resume_epoch, epochs):
        last_epoch = epoch
        dataset_epoch = base_dataset.shuffle(buffer_size=config.SHUFFLE_BUFFER, seed=42 + epoch)
        dataset_epoch = dataset_epoch.with_format("torch")
        if epoch == resume_epoch and start_position > 0:
            dataset_epoch = dataset_epoch.skip(start_position)
        loader = DataLoader(dataset_epoch, batch_size=config.BATCH_SIZE, shuffle=False)
        for step, batch in enumerate(loader):
            last_step = step

            # Get the input IDs and attention mask, and move them to the GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Next-token prediction
            # input = Hello, Wor
            # target = llo, World
            inputs = input_ids[:, :-1] # everything from the first token except the last
            targets = input_ids[:, 1:] # everything from the second token onward

            # Preview the first sample to confirm tokenization/targets.
            if debug_level >= 2 and not printed_debug_sample:
                input_preview = inputs[0].tolist()
                target_preview = targets[0].tolist()
                print(f"Input tokens: {input_preview}")
                print(f"Target tokens: {target_preview}")
                print(f"Input text: {tokenizer.decode(input_preview)}")
                print(f"Target text: {tokenizer.decode(target_preview)}")
                printed_debug_sample = True

            # Dump decoded inputs for every sample in the batch.
            if debug_level >= 6:
                for sample_index, sample_ids in enumerate(input_ids.tolist()):
                    decoded = tokenizer.decode(sample_ids)
                    print(f"Encoded input {sample_index}: {decoded}")

            # Clear accumulated gradients from the previous step (which torch does automatically otherwise)
            optimizer.zero_grad()

            # Forward pass
            logits = model(inputs, attention_mask=attention_mask[:, :-1])

            # Compute (average) loss of the predicted next tokens and apply backpropagation.
            # reshape to (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
            loss = lossFn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()

            # Clip gradients to stabilize training (especially for larger batches).
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights, then advance the learning-rate schedule.
            optimizer.step()
            scheduler.step()

            # Log progress and plot loss history
            token_count = attention_mask[:, 1:].sum().item()
            remaining_tokens = max(target_tokens - progress.total_tokens, 0)
            progress.tick(
                loss.item(),
                input_ids.size(0),
                token_count,
                optimizer.param_groups[0]["lr"],
                epoch,
                step,
                remaining_tokens=remaining_tokens,
            )
            current_position += input_ids.size(0)
            now = time.time()
            ckpt_interval = config.CHECKPOINT_WARMUP_SECS if (now - last_ckpt_time) < config.WARMUP_WINDOW_SECS else config.CHECKPOINT_INTERVAL_SECS
            if now - last_ckpt_time >= ckpt_interval:
                checkpointer.save_latest(
                    epoch,
                    step,
                    progress.global_step,
                    current_position,
                    progress.total_tokens,
                )
                last_ckpt_time = now
        start_position = 0
        current_position = 0

except KeyboardInterrupt:
    # Save a checkpoint so training can resume from the last completed step.
    print("Interrupted: saving checkpoint...")
    checkpointer.save_latest(
        last_epoch,
        last_step,
        progress.global_step,
        current_position,
        progress.total_tokens,
    )
    print("Interrupted: checkpoint saved, exiting.")


