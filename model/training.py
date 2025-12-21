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
        config.apply_overrides(checkpoint_state["config"])

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
from loader import ShardedBatchLoader, build_chunking_tokenizer, build_tokenizer

# Download shards on demand and shuffle within each shard.
set_verbosity_warning()
enable_progress_bar()

# do or not do chunking of the input text, instead of truncating.
if False:
    max_len = context_len
    stride = context_len//4  # overlap; set to 0 for no overlap

    tokenizer.disable_truncation()
    tokenizer.disable_padding()

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
    tokenizer_batch = build_tokenizer(tokenizer)

# Tokenize the dataset
tuned_batch_size = find_max_batch_size(
    model,
    vocab_size=tokenizer.get_vocab_size(),
    seq_len=context_len,
    device=device,
    start=config.BATCH_SIZE,
)
batch_size = tuned_batch_size or config.BATCH_SIZE
print(f"Tuned batch_size={batch_size}")
config.print_training_hyperparams(
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
import time

# Set up optimizer, learning-rate scheduler, and loss function
epochs = 1 # epochs between 1 and 3 are usually sufficient for good results, rather 1 than 3.
estimated_total_samples = sharded_loader.estimate_total_samples()
steps_per_epoch = math.ceil(estimated_total_samples / batch_size)
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

