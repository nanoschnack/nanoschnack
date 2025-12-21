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
from config import (
    BATCH_SIZE,
    CHECKPOINT_INTERVAL_SECS,
    CHECKPOINT_WARMUP_SECS,
    CONTEXT_LEN,
    EMBED_SIZE,
    HIDDEN_SIZE,
    LEARNING_RATE,
    LOG_INTERVAL_SECS,
    MAX_NEW_TOKENS,
    NUM_HEADS,
    NUM_LAYERS,
    PLOT_INTERVAL_SECS,
    PLOT_WARMUP_SECS,
    TEMPERATURE,
    TOP_K,
    WARMUP_PCT,
    WARMUP_WINDOW_SECS,
    print_training_hyperparams,
)

# Resolve model paths so relative data/checkpoint locations are stable.
try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths
model_dir, data_dir, checkpoint_dir = setup_paths()

# Load checkpoint config if present to keep model sizes consistent.
import torch
checkpoint_path = checkpoint_dir / "latest.pt"
checkpoint_config = {}
checkpoint_state = None
if checkpoint_path.exists():
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint_state, dict):
        checkpoint_config.update(checkpoint_state.get("config", {}))
        state_dict = checkpoint_state.get("model", checkpoint_state)
    else:
        state_dict = checkpoint_state
    if "position_embedding.weight" in state_dict:
        checkpoint_config["context_len"] = state_dict["position_embedding.weight"].shape[0]
    if "token_embedding.weight" in state_dict:
        checkpoint_config["embed_size"] = state_dict["token_embedding.weight"].shape[1]

context_len = checkpoint_config.get("context_len", CONTEXT_LEN)
embed_size = checkpoint_config.get("embed_size", EMBED_SIZE)
num_layers = checkpoint_config.get("num_layers", NUM_LAYERS)
num_heads = checkpoint_config.get("num_heads", NUM_HEADS)
hidden_size = checkpoint_config.get("hidden_size", HIDDEN_SIZE)

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
from loader import ShardedBatchLoader

# Download shards on demand and shuffle within each shard.
set_verbosity_warning()
enable_progress_bar()

# do or not do chunking of the input text, instead of truncating.
if False:
    max_len = context_len
    stride = context_len//4  # overlap; set to 0 for no overlap

    tokenizer.disable_truncation()
    tokenizer.disable_padding()

    # Split long sequences into fixed windows, optionally with overlap.
    def chunk_ids(ids, max_len, stride):
        if len(ids) == 0:
            return []
        step = max_len - stride
        chunks = []
        for start in range(0, len(ids), step):
            chunk = ids[start:start + max_len]
            if len(chunk) == 0:
                continue
            if len(chunk) < max_len:
                chunk = chunk + [pad_id] * (max_len - len(chunk))
            chunks.append(chunk)
            if start + max_len >= len(ids):
                break
        return chunks

    def tokenizer_batch(batch):
        input_ids = []
        attention_mask = [] # marks real tokens (1) vs padding (0)
        for text in batch["text"]:
            ids = tokenizer.encode(text).ids
            for chunk in chunk_ids(ids, max_len=max_len,
                                   stride=stride):
                input_ids.append(chunk)
                attention_mask.append([1 if t != pad_id else 0 for t
                                       in chunk])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
else:
    # Enable truncation and padding
    tokenizer.enable_truncation(max_length=context_len)
    tokenizer.enable_padding(length=context_len, pad_id=pad_id, pad_token="[PAD]")

    # Wrap Hugging Face tokenizer for batch processing
    def tokenizer_batch(batch):
        token_batch = tokenizer.encode_batch(batch["text"])
        return {
            "input_ids": [e.ids for e in token_batch],
            "attention_mask": [e.attention_mask for e in token_batch], # marks real tokens (1) vs padding (0)
        }

# Tokenize the dataset
tuned_batch_size = find_max_batch_size(
    model,
    vocab_size=tokenizer.get_vocab_size(),
    seq_len=context_len,
    device=device,
    start=BATCH_SIZE,
)
batch_size = tuned_batch_size or BATCH_SIZE
print(f"Tuned batch_size={batch_size}")
print_training_hyperparams(
    model,
    context_len=context_len,
    embed_size=embed_size,
    num_layers=num_layers,
    num_heads=num_heads,
    hidden_size=hidden_size,
    batch_size=batch_size,
)
sharded_loader = ShardedBatchLoader(
    repo_id="arnomatic/german-wikipedia-clean-no-lists",
    data_dir=data_dir,
    tokenizer_batch=tokenizer_batch,
    batch_size=batch_size,
    seed=42,
)
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
import torch
import time

# Set up optimizer, learning-rate scheduler, and loss function
epochs = 1 # epochs between 1 and 3 are usually sufficient for good results, rather 1 than 3.
estimated_total_samples = sharded_loader.estimate_total_samples()
steps_per_epoch = math.ceil(estimated_total_samples / batch_size)
total_steps = steps_per_epoch * epochs
print(f"Estimated steps per epoch: {steps_per_epoch} (total {total_steps}).", flush=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = build_warmup_cosine(optimizer, total_steps, WARMUP_PCT)
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
    log_interval=LOG_INTERVAL_SECS,
    warmup_plot_interval=PLOT_WARMUP_SECS,
    plot_interval=PLOT_INTERVAL_SECS,
    warmup_window_secs=WARMUP_WINDOW_SECS,
)


last_epoch = resume_epoch
last_step = resume_step
current_position = resume_position
total_samples = total_samples

# Enable preview output when NANOSCHNACK_DEBUG_SAMPLE is set.
debug_sample = os.getenv("NANOSCHNACK_DEBUG_SAMPLE")
printed_debug_sample = False
try:
    print("Starting training loop...", flush=True)
    for epoch in range(resume_epoch, epochs):
        last_epoch = epoch
        loader = sharded_loader.iter_batches(start_position=current_position)
        for step, (batch, current_position, shard_index, shard_len) in enumerate(loader):
            last_step = step

            # Get the input IDs and attention mask, and move them to the GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Next-token prediction
            inputs = input_ids[:, :-1] # everything from the first token except the last
            targets = input_ids[:, 1:] # everything from the second token onward

            # Preview the first sample to confirm tokenization/targets.
            if debug_sample and not printed_debug_sample:
                input_preview = inputs[0].tolist()
                target_preview = targets[0].tolist()
                print(f"Input tokens: {input_preview}")
                print(f"Target tokens: {target_preview}")
                print(f"Input text: {tokenizer.decode(input_preview)}")
                print(f"Target text: {tokenizer.decode(target_preview)}")
                printed_debug_sample = True

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
                shard_count=sharded_loader.num_shards,
                shard_len=shard_len,
                remaining_samples=remaining_samples,
            )
            total_samples += input_ids.size(0)
            now = time.time()
            ckpt_interval = CHECKPOINT_WARMUP_SECS if (now - last_ckpt_time) < WARMUP_WINDOW_SECS else CHECKPOINT_INTERVAL_SECS
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

