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
# ## Setup Devices
#
# - Verify that MPS is available (for Apple Silicon GPUs).

# %%
import contextlib
import os
import torch

from device import device_info, pick_device, print_ddp_info, print_device_info, print_sdpa_info

# Setup distributed data parallel (DDP)
ddp_rank = int(os.getenv("RANK", "0"))
ddp_world_size = int(os.getenv("WORLD_SIZE", "1"))
ddp_local_rank = int(os.getenv("LOCAL_RANK", "0"))
ddp_master_addr = os.getenv("MASTER_ADDR", None)
ddp_master_port = os.getenv("MASTER_PORT", None)
ddp_enabled = ddp_world_size > 1
is_master = ddp_rank == 0
if ddp_enabled:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requested without CUDA availability.")
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")

# Select the device for this process.
device = pick_device(ddp_local_rank)
info = device_info(device)
if is_master:
    print_device_info(info)
    if ddp_enabled:
        print_ddp_info(ddp_rank, ddp_world_size, ddp_local_rank, ddp_master_addr, ddp_master_port)
    print_sdpa_info()

# Switch to TF32 for 8x speedup on supported hardware, and good enough for LLM training.
torch.set_float32_matmul_precision("high")


# %% [markdown]
# ## Loading a tokenizer with Hugging Face's tokenizer library
#
# - Compare: https://github.com/huggingface/tokenizers
# - Tiktokenizer: https://tiktokenizer.vercel.app/?model=gpt2

# %%
from tokenizer import PAD_TOKEN, load_tokenizer, print_vocab_alignment
tokenizer = load_tokenizer()
if is_master:
    print_vocab_alignment(tokenizer)


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
tokenizer.add_special_tokens([PAD_TOKEN])
pad_id = tokenizer.token_to_id(PAD_TOKEN)

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
config.BATCH_SIZE = config.align_micro_batch_size(
    config.BATCH_SIZE,
    config.MACRO_BATCH_SIZE,
)

# Compile the model for faster training.
if device.type == "cuda":
    print("Compiling the model for faster training...") if is_master else None
    model = torch.compile(model)

param_count, quantization = config.model_info(model)
if is_master:
    config.print_training_hyperparams(param_count=param_count, quantization=quantization)

# %% [markdown]
# ## Create vizualization of the model

# %%
# Visualize the model graph only in interactive notebooks.
try:
    from IPython import get_ipython
    is_notebook = get_ipython() is not None and "IPKernelApp" in get_ipython().config
except Exception:
    is_notebook = False

if is_notebook:
    from torchviz import make_dot

    vocab_size = tokenizer.get_vocab_size()
    viz_batch_size = min(config.BATCH_SIZE, 2)
    viz_context_len = min(config.CONTEXT_LEN, 16)
    x = torch.randint(
        0,
        vocab_size,
        (viz_batch_size, viz_context_len),
        device=device,
        dtype=torch.long,
    )
    y = model(x)

    make_dot(y, params=dict(model.named_parameters()))


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
from resume import build_resume_state, is_resume_exhausted, normalize_resume_rows
import math

# Download shards on demand and shuffle within each dataset.
set_verbosity_warning()
enable_progress_bar()

# Cache dataset specs for reuse across steps.
dataset_specs = parse_dataset_specs(config.DATASET_SPECS)

# Track total rows per dataset for resume validation.
total_rows_by_spec = {}
estimated_total_tokens = 0

print("Datasets:") if is_master else None
for dataset_index, spec in enumerate(dataset_specs):
    raw_dataset = load_dataset_from_spec(
        spec,
        cache_dir=data_dir,
        streaming=True,
    )
    if ddp_enabled:
        raw_dataset = raw_dataset.shard(num_shards=ddp_world_size, index=ddp_rank)
    total_rows = resolve_total_rows(raw_dataset, spec)
    total_rows_by_spec[spec["spec"]] = total_rows
    if total_rows is None:
        raise ValueError("Dataset split metadata missing num_examples for token estimate.")
    token_estimator = TokenEstimator(
        tokenizer,
        text_key=spec["text_key"],
    )
    avg_tokens, est_total_tokens = token_estimator.estimate_streaming(raw_dataset, total_rows)
    estimated_total_tokens += est_total_tokens
    print(f"    {spec}: avg_tokens={avg_tokens:.1f}, est_tokens={est_total_tokens}") if is_master else None

# Resolve model size for token budgeting.
param_count, _ = config.model_info(model)

# Derive the token cap and epoch count from the configured max-training factor.
max_tokens = int(param_count * config.MAX_TRAINING_FACTOR) if config.MAX_TRAINING_FACTOR > 0 else 0
target_tokens = max_tokens or estimated_total_tokens
target_epochs = 1
if max_tokens and estimated_total_tokens > 0:
    target_epochs = max(1, math.ceil(target_tokens / estimated_total_tokens))
if config.MACRO_BATCH_SIZE % config.BATCH_SIZE != 0:
    raise ValueError("MACRO_BATCH_SIZE must be divisible by BATCH_SIZE.")
tokens_per_step = config.MACRO_BATCH_SIZE * (config.CONTEXT_LEN - 1)
dataset_steps = math.ceil(estimated_total_tokens / tokens_per_step)
if is_master:
    print(f"    Dataset estimate: steps={dataset_steps:,} tokens={estimated_total_tokens:,} tokens_per_step={tokens_per_step:,}")
    print(f"    Target: epochs={target_epochs:,} target_tokens={target_tokens:,} (factor {config.MAX_TRAINING_FACTOR} of model size {param_count:,})")


# %% [markdown]
# ## Progress and Plotting

# %%
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


# %% [markdown]
# ## Load the previous Checkpoint

# %%
from plot import ascii_loss_plot
from chat import generate_reply_stream
from progress import ProgressLogger
from checkpointer import Checkpointer
from scheduler import build_warmup_cosine_tokens
from torch.utils.data import DataLoader
import itertools
import os
import signal
import time

# Set up optimizer, learning-rate scheduler, and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
scheduler = build_warmup_cosine_tokens(optimizer, target_tokens, config.WARMUP_PCT)
lossFn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
checkpointer = Checkpointer(checkpoint_dir, model, optimizer, scheduler, device=device)

# Load the latest checkpoint if available.
resume_epoch, resume_step, global_step, resume_sample_index, resume_tokens, resume_state = checkpointer.load_latest()

# Align the scheduler with the resumed token count.
if resume_tokens:
    scheduler.last_epoch = resume_tokens # we misuse token's epoch count for tokens
    for group, base_lr, lr_lambda in zip(optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
        group["lr"] = base_lr * lr_lambda(resume_tokens)

# Normalize resume state into per-spec row offsets.
# Keep offsets from the checkpoint, even for specs not active in this run.
# Ensure current specs always have a default offset for safe lookups.
# This drives shard/row skipping during resume and checkpointing.
resume_rows = normalize_resume_rows(resume_state, dataset_specs)

# Convert global resume offsets to per-rank offsets for sharded streams.
if ddp_enabled:
    def _per_rank_row_offset(row_offset):
        return 0 if row_offset <= ddp_rank else (row_offset - ddp_rank + ddp_world_size - 1) // ddp_world_size
    resume_rows = {spec_key: _per_rank_row_offset(offset) for spec_key, offset in resume_rows.items()}

# Track row offsets for shard-aware resume.
source_row_counts = dict(resume_rows)

# Report when resuming via the legacy sample index.
use_row_resume = any(resume_rows.get(spec["spec"], 0) > 0 for spec in dataset_specs)
# Track the dataset position for skipping/checkpointing when row offsets are unavailable.
loader_skip_samples = 0 if use_row_resume else resume_sample_index
if resume_sample_index > 0 and not use_row_resume:
    print(f"Resume rows unavailable; falling back to linear sample skip ({resume_sample_index}).")

# Build packed datasets per source with row-offset resumes.
# Resolve shard-aware resume plans, then stream from the right shard/offset.
# Pack each source into fixed-length token blocks with source IDs.
# Interleave happens later, so each dataset is prepared independently.
# Row offsets are tracked for checkpoint-safe restarts.
packed_datasets = []
for dataset_index, spec in enumerate(dataset_specs):
    row_offset = resume_rows.get(spec["spec"], 0)
    # Skip datasets that are already fully consumed by resume offsets.
    total_rows = total_rows_by_spec.get(spec["spec"])
    if is_resume_exhausted(row_offset, total_rows):
        print(f"Skipping exhausted dataset {spec['spec']}: row_offset {row_offset} >= total_rows {total_rows}") if is_master else None
        continue
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
    if ddp_enabled:
        raw_streaming = raw_streaming.shard(num_shards=ddp_world_size, index=ddp_rank)
    if row_offset > 0:
        if data_files is None:
            print(f"Resume rows (linear): {spec['spec']} -> {row_offset}") if is_master else None
            raw_streaming = raw_streaming.skip(row_offset)
        else:
            print(f"Resume rows: {spec['spec']} -> {shard_label} +{in_shard_offset}") if is_master else None
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

if not packed_datasets:
    raise ValueError("All datasets exhausted after resume; check DATASET_SPECS.")

base_dataset = build_interleaved_dataset(packed_datasets, seed=42)
if is_master:
    print(f"Packed dataset ready ({len(packed_datasets)} sources).", flush=True)

# %% [markdown]
# ## Run the Training

# %%
last_ckpt_time = time.time()

# Track current counters for checkpointing and interrupts.
current_epoch = resume_epoch
current_step = resume_step
current_sample_index = resume_sample_index
current_micro_step = 0
micro_steps = config.MACRO_BATCH_SIZE // config.BATCH_SIZE
micro_loss_total = 0.0
micro_token_total = 0
micro_sample_total = 0

# Initialize the progress logger to display training progress and loss
progress = ProgressLogger(
    plot_with_completion,
    start_global_step=global_step,
    start_total_samples=resume_sample_index,
    start_total_tokens=resume_tokens,
    warmup_plot_interval=config.PLOT_WARMUP_SECS,
    plot_interval=config.PLOT_INTERVAL_SECS,
    warmup_window_secs=config.WARMUP_WINDOW_SECS,
    estimated_total_tokens=target_tokens,
)

# Enable debug output with DEBUG levels.
debug_level = int(os.getenv("DEBUG", "0"))
printed_debug_sample = False

# Track SIGINT so we can checkpoint after a safe step.
stop_requested = False
def _request_stop(signum, frame):
    # Record interrupt without raising inside the signal handler.
    print("Interrupted: saving checkpoint...") if is_master else None
    global stop_requested
    stop_requested = True
signal.signal(signal.SIGINT, _request_stop)

print("Starting training loop...", flush=True) if is_master else None
for current_epoch in itertools.count(resume_epoch):
    # Reset row counters at epoch boundaries beyond the resume epoch.
    if current_epoch != resume_epoch:
        for spec in dataset_specs:
            source_row_counts[spec["spec"]] = 0
    dataset_epoch = base_dataset.shuffle(buffer_size=config.SHUFFLE_BUFFER, seed=42 + current_epoch)
    dataset_epoch = dataset_epoch.with_format("torch")
    if current_epoch == resume_epoch and loader_skip_samples > 0:
        dataset_epoch = dataset_epoch.skip(loader_skip_samples)
    loader = DataLoader(dataset_epoch, batch_size=config.BATCH_SIZE, shuffle=False)

    for current_step, batch in enumerate(loader):
        # Move batch tensors to the device and prepare an optional attention mask.
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        attn_mask = None
        if attention_mask is not None and not attention_mask.all():
            attn_mask = attention_mask[:, :-1].to(device)

        # Build next-token prediction pairs.
        inputs = input_ids[:, :-1].to(device) # everything from the first token except the last
        targets = input_ids[:, 1:].to(device) # everything from the second token onward

        # Preview tokenization outputs for debugging.
        if debug_level >= 2 and not printed_debug_sample and is_master:
            input_preview = inputs[0].tolist()
            target_preview = targets[0].tolist()
            print(f"Input tokens: {input_preview}")
            print(f"Target tokens: {target_preview}")
            print(f"Input text: {tokenizer.decode(input_preview)}")
            print(f"Target text: {tokenizer.decode(target_preview)}")
            printed_debug_sample = True

        # Clear accumulated gradients before the first micro step in the macro batch.
        if current_micro_step == 0:
            optimizer.zero_grad()
            micro_loss_total = 0.0
            micro_token_total = 0
            micro_sample_total = 0

        # Run the forward pass with autocast and compute loss.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else contextlib.nullcontext():
            logits = model(inputs, attention_mask=attn_mask)
            loss = lossFn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        # Scale loss for gradient accumulation.
        scaled_loss = loss / micro_steps
        scaled_loss.backward()

        # Micro step bookkeeping.
        micro_loss_total += loss.item()
        token_count = attention_mask[:, 1:].sum().item()
        micro_token_total += token_count
        micro_sample_total += input_ids.size(0)

        # Advance per-source row counters for resume safety.
        row_counts = batch["row_count"].tolist()
        source_ids = batch["source_id"].tolist()
        for source_id, row_count in zip(source_ids, row_counts):
            if row_count:
                spec_key = dataset_specs[int(source_id)]["spec"]
                source_row_counts[spec_key] += int(row_count)

        # Update checkpoint counters and save when needed.
        current_sample_index += input_ids.size(0)
        if current_micro_step != micro_steps - 1:
            current_micro_step += 1
            continue

        # Log progress and plot loss history.
        next_total_tokens = progress.total_tokens + micro_token_total
        remaining_tokens = max(target_tokens - next_total_tokens, 0)

        # Average the micro loss across ranks for consistent logging.
        logged_loss_total = micro_loss_total
        if ddp_enabled:
            loss_tensor = torch.tensor(micro_loss_total, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            logged_loss_total = loss_tensor.item() / ddp_world_size

        progress.tick(
            logged_loss_total / micro_steps,
            micro_sample_total,
            micro_token_total,
            optimizer.param_groups[0]["lr"],
            current_epoch,
            current_step,
            remaining_tokens=remaining_tokens,
        )

        # Print a target decode alongside log output when debugging.
        if debug_level >= 1 and is_master:
            target_ids = targets[-1]
            if attention_mask is not None:
                mask = attention_mask[-1, 1:].bool()
                target_ids = target_ids[mask]
            decoded_target = tokenizer.decode(target_ids.tolist())
            print(f"Learn target: {decoded_target}")

        # Apply gradient clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Apply the optimizer step.
        optimizer.step()

        # Apply the optimizer step and advance the token scheduler.
        scheduler.last_epoch = next_total_tokens - 1
        scheduler.step()

        now = time.time()
        ckpt_interval = config.CHECKPOINT_WARMUP_SECS if (now - last_ckpt_time) < config.WARMUP_WINDOW_SECS else config.CHECKPOINT_INTERVAL_SECS
        should_checkpoint = (now - last_ckpt_time >= ckpt_interval) or stop_requested
        if should_checkpoint:
            # Build the resume state for the checkpoint.
            resume_state = build_resume_state(source_row_counts, dataset_specs)

            # Aggregate per-rank row counts for global resume offsets.
            if ddp_enabled:
                spec_keys = [spec["spec"] for spec in dataset_specs]
                counts_tensor = torch.tensor(
                    [source_row_counts.get(spec_key, 0) for spec_key in spec_keys],
                    dtype=torch.long,
                    device=device,
                )
                dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
                if is_master:
                    global_counts = dict(source_row_counts)
                    for spec_key, value in zip(spec_keys, counts_tensor.tolist()):
                        global_counts[spec_key] = int(value)
                    resume_state = build_resume_state(global_counts, dataset_specs)

            # Persist checkpoints only from the master process.
            if is_master:
                checkpointer.save_latest(
                    current_epoch,
                    current_step,
                    progress.global_step,
                    current_sample_index,
                    progress.total_tokens,
                    resume_state=resume_state,
                )
            last_ckpt_time = now

        # Exit after the current step if SIGINT was requested.
        if stop_requested:
            break

        current_micro_step = 0
    # Reset sample skip counter after the first epoch; partial macro batches spill to next epoch.
    loader_skip_samples = 0
    current_sample_index = 0

    if stop_requested:
        break



