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
    ChunkEstimator,
    MultiShardedBatchLoader,
    ShardedBatchLoader,
    build_chunking_tokenizer,
    build_tokenizer,
)

# Download shards on demand and shuffle within each shard.
set_verbosity_warning()
enable_progress_bar()

# do or not do chunking of the input text, instead of truncating.
if True:
    max_len = context_len
    stride = context_len//4  # overlap; set to 0 for no overlap

    tokenizer.no_padding()
    tokenizer.no_truncation()

    # Estimate chunk counts from a sample; chunking stays in-memory.
    estimator = ChunkEstimator(
        tokenizer,
        max_len=max_len,
        stride=stride,
    )

    # Build a tokenizer that chunks long sequences on the fly.
    tokenizer_batch = build_chunking_tokenizer(
        tokenizer,
        pad_id=pad_id,
        max_len=max_len,
        stride=stride,
    )
else:
    # Enable truncation and padding
    tokenizer.enable_truncation(max_length=context_len)
    tokenizer.enable_padding(length=context_len, pad_id=pad_id, pad_token="[PAD]")

    # Wrap Hugging Face tokenizer for batch processing
    estimator = None
    tokenizer_batch = build_tokenizer(tokenizer)

# Now with the tokenizer derive training parameters like batch size.
tuned_batch_size = find_max_batch_size(
    model,
    vocab_size=tokenizer.get_vocab_size(),
    seq_len=context_len,
    device=device,
    start=config.BATCH_SIZE,
)
if tuned_batch_size:
    config.BATCH_SIZE = tuned_batch_size
print(f"Tuned batch_size={config.BATCH_SIZE}")
param_count, quantization = config.model_info(model)
config.print_training_hyperparams(param_count=param_count, quantization=quantization)

# Build one sharded loader per dataset.
loaders = []
repo_ids = [
    "arnomatic/german-wikipedia-clean-no-lists",
    "PatrickHaller/fineweb-2-de-1B",
]
for repo_id in repo_ids:
    loaders.append(
        ShardedBatchLoader(
            repo_id=repo_id,
            data_dir=data_dir,
            tokenizer_batch=tokenizer_batch,
            batch_size=config.BATCH_SIZE,
            seed=42,
        )
    )

# Interleave datasets round-robin.
sharded_loader = MultiShardedBatchLoader(loaders)
print(f"Sharded loader ready ({sharded_loader.num_shards} shards).", flush=True)

# %% [markdown]
# ## Run the Training

# %%
from plot import ascii_loss_plot
from progress import ProgressLogger
from checkpointer import Checkpointer
from scheduler import build_warmup_cosine
import math
import os
import time

# Set up optimizer, learning-rate scheduler, and loss function
epochs = 1 # epochs between 1 and 3 are usually sufficient for good results, rather 1 than 3.
estimated_total_samples = sharded_loader.estimate_total_samples()

# Report chunk estimates when chunking is enabled.
if estimator is not None:
    print("Estimating chunks from first shard...", flush=True)
    estimated_total_samples = 0

    for dataset_index, loader in enumerate(sharded_loader.loaders):
        raw_shard = loader.shards.load_shard(0)
        avg_chunks, est_total = estimator.estimate_dataset(raw_shard)
        loader.set_estimated_shard_len(est_total)
        estimated_total_samples += est_total * loader.num_shards
        print(
            f"Dataset {dataset_index + 1}/{sharded_loader.num_datasets} "
            f"({loader.shards.repo_id}): avg_chunks={avg_chunks:.2f}, "
            f"est_total={est_total}"
        )
print(f"Estimated total samples (sentences) across datasets: {estimated_total_samples}", flush=True)
print(f"Estimated total tokens across datasets: {estimated_total_samples * context_len}", flush=True)

# Chunk-aware estimates are already applied when the estimator is set.
steps_per_epoch = math.ceil(estimated_total_samples / config.BATCH_SIZE)
total_steps = steps_per_epoch * epochs
print(f"Estimated steps per epoch: {steps_per_epoch} (total {total_steps}).", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
scheduler = build_warmup_cosine(optimizer, total_steps, config.WARMUP_PCT)
lossFn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

# The checkpointer will save and load model/optimizer/scheduler states to/from disk.
checkpointer = Checkpointer(checkpoint_dir, model, optimizer, scheduler, device=device)
resume_epoch, resume_step, global_step, resume_position, total_samples, resume_tokens = checkpointer.load_latest()

last_ckpt_time = time.time()

# Initialize the progress logger to display training progress and loss
progress = ProgressLogger(
    ascii_loss_plot,
    start_global_step=global_step,
    start_total_samples=total_samples,
    start_total_tokens=resume_tokens,
    log_interval=config.LOG_INTERVAL_SECS,
    warmup_plot_interval=config.PLOT_WARMUP_SECS,
    plot_interval=config.PLOT_INTERVAL_SECS,
    warmup_window_secs=config.WARMUP_WINDOW_SECS,
)


last_epoch = resume_epoch
last_step = resume_step
if not isinstance(resume_position, list):
    resume_position = [(0, 0) for _ in range(sharded_loader.num_datasets)]
current_position = resume_position

# Enable debug output with DEBUG levels.
debug_level = int(os.getenv("DEBUG", "0"))
printed_debug_sample = False
try:
    print("Starting training loop...", flush=True)
    for epoch in range(resume_epoch, epochs):
        last_epoch = epoch
        loader = sharded_loader.iter_batches(start_positions=current_position)
        for step, batch_info in enumerate(loader):
            last_step = step

            # Unpack per-dataset batches.
            batch, current_position, dataset_index, shard_index, shard_len = batch_info
            shard_count = sharded_loader.loaders[dataset_index].num_shards

            # Get the input IDs and attention mask, and move them to the GPU
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(device)

            # Next-token prediction
            # input = Hello, Wor
            # target = llo, World
            inputs = input_ids[:, :-1].to(device) # everything from the first token except the last
            targets = input_ids[:, 1:].to(device) # everything from the second token onward

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
            remaining_samples = max(estimated_total_samples * epochs - total_samples, 0)
            progress.tick(
                loss.item(),
                input_ids.size(0),
                token_count,
                optimizer.param_groups[0]["lr"],
                epoch,
                step,
                shard_index=shard_index,
                shard_count=shard_count,
                shard_len=shard_len,
                remaining_samples=remaining_samples,
            )
            total_samples += input_ids.size(0)
            now = time.time()
            ckpt_interval = config.CHECKPOINT_WARMUP_SECS if (now - last_ckpt_time) < config.WARMUP_WINDOW_SECS else config.CHECKPOINT_INTERVAL_SECS
            if now - last_ckpt_time >= ckpt_interval:
                checkpointer.save_latest(
                    epoch,
                    step,
                    progress.global_step,
                    current_position,
                    total_samples,
                    progress.total_tokens,
                )
                last_ckpt_time = now
        current_position = (0, 0)

except KeyboardInterrupt:
    # Save a checkpoint so training can resume from the last completed step.
    print("Interrupted: saving checkpoint...")
    checkpointer.save_latest(
        last_epoch,
        last_step,
        progress.global_step,
        current_position,
        total_samples,
        progress.total_tokens,
    )
    print("Interrupted: checkpoint saved, exiting.")

