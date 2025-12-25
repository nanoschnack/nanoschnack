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
import contextlib
import torch
from device import device_info, pick_device, print_device_info

device = pick_device()
info = device_info(device)
print_device_info(info)
# Report SDPA kernel availability for attention debugging.
print("Performance:")
print(f"  Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"  Mem-efficient SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"  Math SDP enabled: {torch.backends.cuda.math_sdp_enabled()}")
print("  SDPA kernel selection: set TORCH_LOGS=attention")

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

# Compile the model for faster training.
if device.type == "cuda":
    print("Compiling the model for faster training...")
    model = torch.compile(model)

param_count, quantization = config.model_info(model)
config.print_training_hyperparams(param_count=param_count, quantization=quantization)


# %% [markdown]
# ## Load the Training Data

# %%
from datasets.utils.logging import enable_progress_bar, set_verbosity_warning
from loader import (
    TokenEstimator,
    build_interleaved_dataset,
    build_packed_dataset,
    load_dataset_from_spec,
    parse_dataset_specs,
    resolve_resume_plan,
    resolve_total_rows,
    dataset_label,
)
from resume import build_resume_state, normalize_resume_rows
import math

# Download shards on demand and shuffle within each dataset.
set_verbosity_warning()
enable_progress_bar()

# Cache dataset specs for reuse across steps.
dataset_specs = parse_dataset_specs(config.DATASET_SPECS)
estimated_total_tokens = 0

print("Estimating tokens from dataset samples...", flush=True)
for dataset_index, spec in enumerate(dataset_specs):
    raw_dataset = load_dataset_from_spec(
        spec,
        cache_dir=data_dir,
        streaming=True,
    )
    total_rows = resolve_total_rows(raw_dataset, spec)
    if total_rows is None:
        raise ValueError("Dataset split metadata missing num_examples for token estimate.")
    token_estimator = TokenEstimator(tokenizer, text_key=spec["text_key"])
    avg_tokens, est_total_tokens = token_estimator.estimate_streaming(raw_dataset, total_rows)
    estimated_total_tokens += est_total_tokens
    print(
        f"Dataset {dataset_index + 1}/{len(dataset_specs)} "
        f"({dataset_label(spec)}): avg_tokens={avg_tokens:.1f}, "
        f"est_tokens={est_total_tokens}"
    )

# Resolve model size for token budgeting.
param_count, _ = config.model_info(model)

# Derive the token cap and epoch count from the configured max-training factor.
max_tokens = int(param_count * config.MAX_TRAINING_FACTOR) if config.MAX_TRAINING_FACTOR > 0 else 0
target_tokens = max_tokens or estimated_total_tokens
if max_tokens and estimated_total_tokens > 0:
    epochs = max(1, math.ceil(target_tokens / estimated_total_tokens))
tokens_per_step = config.BATCH_SIZE * (config.CONTEXT_LEN - 1)
dataset_steps = math.ceil(estimated_total_tokens / tokens_per_step)
print(
    f"Dataset estimate: steps={dataset_steps:,} tokens={estimated_total_tokens:,} "
    f"tokens_per_step={tokens_per_step:,}",
    flush=True,
)
print(
    f"Target:          epochs={epochs:,} target_tokens={target_tokens:,} "
    f"(factor {config.MAX_TRAINING_FACTOR} of model size {param_count:,})",
    flush=True,
)



# %% [markdown]
# ## Run the Training

# %%
from plot import ascii_loss_plot
from chat import generate_reply_stream
from progress import ProgressLogger
from checkpointer import Checkpointer
from scheduler import build_warmup_cosine_tokens
from torch.utils.data import DataLoader
import os
import signal
import time

def plot_with_completion(points):
    # Render the loss plot first so completion failures don't block logs.
    chart = ascii_loss_plot(points)

    # Append the configured completion snapshot.
    was_training = model.training
    if was_training:
        model.eval()
    try:
        reply_parts = []
        for token in generate_reply_stream(
            model,
            tokenizer,
            config.PLOT_COMPLETION_PROMPT,
            context_len=config.CONTEXT_LEN,
            max_new_tokens=config.PLOT_COMPLETION_TOKENS,
            temperature=config.TEMPERATURE,
            top_k=config.TOP_K,
            device=device,
        ):
            reply_parts.append(token)
        completion = "".join(reply_parts)
    except Exception as exc:
        completion = f" [generation failed: {exc}]"
    finally:
        if was_training:
            model.train()
    return (
        f"{chart}\n\ncompletion ({config.PLOT_COMPLETION_TOKENS} tokens)\n"
        f"{config.PLOT_COMPLETION_PROMPT}{completion}"
    )

# Set up optimizer, learning-rate scheduler, and loss function
epochs = 1 # epochs between 1 and 3 are usually sufficient for good results, rather 1 than 3.

optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
scheduler = build_warmup_cosine_tokens(optimizer, target_tokens, config.WARMUP_PCT)
lossFn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
# The checkpointer will save and load model/optimizer/scheduler states to/from disk.
checkpointer = Checkpointer(checkpoint_dir, model, optimizer, scheduler, device=device)
resume_epoch, resume_step, global_step, sample_index, resume_tokens = checkpointer.load_latest()
resume_state = checkpointer.resume_state

# Extend the plan when resuming beyond the original epoch budget.
if resume_epoch >= epochs:
    print(
        f"Resume epoch {resume_epoch + 1} exceeds planned epochs {epochs}; extending plan.",
        flush=True,
    )
    epochs = resume_epoch + 1


# Track token counts for the token-based scheduler.
total_tokens_seen = resume_tokens

# Align the scheduler with the resumed token count.
if resume_tokens:
    scheduler.last_epoch = resume_tokens
    for group, base_lr, lr_lambda in zip(
        optimizer.param_groups,
        scheduler.base_lrs,
        scheduler.lr_lambdas,
    ):
        group["lr"] = base_lr * lr_lambda(resume_tokens)

# Normalize resume state into per-spec row offsets.
resume_rows = normalize_resume_rows(resume_state, dataset_specs)

# Report when resuming via the legacy sample index.
use_row_resume = any(resume_rows.get(spec["spec"], 0) > 0 for spec in dataset_specs)
if sample_index > 0 and not use_row_resume:
    print(
        f"Resume rows unavailable; falling back to linear sample skip ({sample_index}).",
        flush=True,
    )

# Build packed datasets per source with row-offset resumes.
packed_datasets = []
for dataset_index, spec in enumerate(dataset_specs):
    row_offset = resume_rows.get(spec["spec"], 0)
    data_files, in_shard_offset, shard_label = resolve_resume_plan(
        spec,
        row_offset,
        cache_dir=data_dir,
    )
    raw_streaming = load_dataset_from_spec(
        spec,
        cache_dir=data_dir,
        streaming=True,
        data_files=data_files,
    )
    if row_offset > 0:
        if data_files is None:
            print(
                f"Resume rows (linear): {spec['spec']} -> {row_offset}",
                flush=True,
            )
            raw_streaming = raw_streaming.skip(row_offset)
        else:
            print(
                f"Resume rows: {spec['spec']} -> {shard_label} +{in_shard_offset}",
                flush=True,
            )
            raw_streaming = raw_streaming.skip(in_shard_offset)
    packed = build_packed_dataset(
        raw_streaming,
        tokenizer=tokenizer,
        block_size=config.CONTEXT_LEN,
        text_key=spec["text_key"],
        pack_batch_size=config.PACK_BATCH_SIZE,
        source_id=dataset_index,
    )
    packed_datasets.append(packed)

base_dataset = build_interleaved_dataset(packed_datasets, seed=42)
print(f"Packed dataset ready ({len(packed_datasets)} sources).", flush=True)

last_ckpt_time = time.time()

# Initialize the progress logger to display training progress and loss
progress = ProgressLogger(
    plot_with_completion,
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
# Resume from the saved sample index when no row offsets are available.
start_position = 0 if use_row_resume else sample_index
current_position = sample_index

# Track row offsets for shard-aware resume.
source_row_counts = dict(resume_rows)

def _build_resume_state():
    # Capture current row offsets per dataset for checkpointing.
    return build_resume_state(source_row_counts, dataset_specs)

# Enable debug output with DEBUG levels.
debug_level = int(os.getenv("DEBUG", "0"))
printed_debug_sample = False

# Track SIGINT so we can checkpoint after a safe step.
stop_requested = {"flag": False}
def _request_stop(signum, frame):
    # Record interrupt without raising inside the signal handler.
    stop_requested["flag"] = True
signal.signal(signal.SIGINT, _request_stop)
try:
    print("Starting training loop...", flush=True)
    for epoch in range(resume_epoch, epochs):
        last_epoch = epoch
        # Reset row counters at epoch boundaries beyond the resume epoch.
        if epoch != resume_epoch:
            for spec in dataset_specs:
                source_row_counts[spec["spec"]] = 0
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
            # Skip the attention mask when all tokens are valid to keep SDPA fast paths.
            attn_mask = None
            if attention_mask is not None and not attention_mask.all():
                attn_mask = attention_mask

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

            # Forward pass with bf16 autocast on CUDA.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext():
                logits = model(inputs, attention_mask=attn_mask[:, :-1] if attn_mask is not None else None)

                # Compute (average) loss of the predicted next tokens and apply backpropagation.
                # reshape to (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
                loss = lossFn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            # Backward pass to compute gradients.
            loss.backward()

            # Clip gradients to stabilize training (especially for larger batches).
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update weights, then advance the learning-rate schedule.
            optimizer.step()

            # Advance the token-based scheduler using observed tokens.
            token_count = attention_mask[:, 1:].sum().item()
            total_tokens_seen += token_count
            scheduler.last_epoch = total_tokens_seen - 1
            scheduler.step()

            # Log progress and plot loss history
            remaining_tokens = max(target_tokens - total_tokens_seen, 0)
            progress.tick(
                loss.item(),
                input_ids.size(0),
                token_count,
                optimizer.param_groups[0]["lr"],
                epoch,
                step,
                remaining_tokens=remaining_tokens,
            )
            # Advance per-source row counters for resume safety.
            row_counts = batch["row_count"].tolist()
            source_ids = batch["source_id"].tolist()
            for source_id, row_count in zip(source_ids, row_counts):
                if row_count:
                    spec_key = dataset_specs[int(source_id)]["spec"]
                    source_row_counts[spec_key] += int(row_count)
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
                    resume_state=_build_resume_state(),
                )
                last_ckpt_time = now

            # Exit after the current step if SIGINT was requested.
            if stop_requested["flag"]:
                raise KeyboardInterrupt
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
        resume_state=_build_resume_state(),
    )
    print("Interrupted: checkpoint saved, exiting.")
