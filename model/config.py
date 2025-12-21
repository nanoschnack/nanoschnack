"""Shared hyperparameters for training and inference."""

import os
import torch

###
### Model architecture
###

def _env_override(name, cast, default):
    value = os.getenv(name)
    if value is None:
        return default
    return cast(value)


def _env_int(name, default):
    return _env_override(name, int, default)


def _env_float(name, default):
    return _env_override(name, float, default)


# Maximum sequence length used to size positional embeddings.
# Keep this aligned between training and inference.
CONTEXT_LEN = _env_int("CONTEXT_LEN", 1024)

# Embedding dimensionality for token and position representations.
# Larger values increase model capacity and compute cost.
EMBED_SIZE = _env_int("EMBED_SIZE", 768)

# Number of Transformer encoder layers in the model.
# Higher values deepen the network and increase training time.
NUM_LAYERS = _env_int("NUM_LAYERS", 12)

# Number of attention heads per Transformer layer.
# Must divide EMBED_SIZE evenly.
NUM_HEADS = _env_int("NUM_HEADS", 8)

# Feed-forward hidden size inside each Transformer layer.
# Often 4x EMBED_SIZE for Transformer blocks.
HIDDEN_SIZE = _env_int("HIDDEN_SIZE", 3072)

###
### Training defaults
###

# Default batch size for training loops.
# Adjust based on device memory.
BATCH_SIZE = _env_int("BATCH_SIZE", 32)

# Default learning rate for the optimizer.
# Tune alongside batch size and scheduler.
LEARNING_RATE = _env_float("LEARNING_RATE", 1e-4)

# Warmup fraction of total training steps for LR ramp-up.
# Use values between 0.01 and 0.05 for small warmups.
WARMUP_PCT = _env_float("WARMUP_PCT", 0.03)

###
### Inference defaults
###

# Default maximum number of tokens to generate per reply.
# Increase for longer responses at higher compute cost.
MAX_NEW_TOKENS = _env_int("MAX_NEW_TOKENS", 128)

# Sampling temperature for inference (0 disables sampling).
# Lower values are more deterministic.
TEMPERATURE = _env_float("TEMPERATURE", 0.8)

# Top-k cutoff for sampling (0 disables top-k filtering).
# Use 0 to sample from the full distribution.
TOP_K = _env_int("TOP_K", 50)

###
### Logging and checkpoint cadence
###

# Checkpoint cadence in seconds after the warmup window.
# Used after the initial warmup period.
CHECKPOINT_INTERVAL_SECS = _env_int("CHECKPOINT_INTERVAL_SECS", 600)

# Checkpoint cadence in seconds during the first 10 minutes.
# Frequent saves help capture early training progress.
CHECKPOINT_WARMUP_SECS = _env_int("CHECKPOINT_WARMUP_SECS", 60)

# Warmup window length in seconds before switching to standard intervals.
# Shared by plotting and checkpoint scheduling.
WARMUP_WINDOW_SECS = _env_int("WARMUP_WINDOW_SECS", 600)

# Plot cadence in seconds after the warmup window.
# Controls how often loss charts are printed.
PLOT_INTERVAL_SECS = _env_int("PLOT_INTERVAL_SECS", 600)

# Plot cadence in seconds during the first 10 minutes.
# Higher frequency helps during the startup phase.
PLOT_WARMUP_SECS = _env_int("PLOT_WARMUP_SECS", 60)

# Log cadence in seconds for progress updates.
# Impacts console verbosity and throughput reporting.
LOG_INTERVAL_SECS = _env_int("LOG_INTERVAL_SECS", 10)


def _format_param_count(count):
    return f"{count:,}"


def _model_param_count(model):
    return sum(param.numel() for param in model.parameters())


def _model_quantization(model):
    quantized_dtypes = {torch.qint8, torch.quint8, torch.qint32}
    dtypes = {param.dtype for param in model.parameters()}
    if dtypes & quantized_dtypes:
        return "int8"
    return "none"


def print_training_hyperparams(
    model=None,
    context_len=None,
    embed_size=None,
    num_layers=None,
    num_heads=None,
    hidden_size=None,
    batch_size=None,
):
    """Print the training-related hyperparameters."""
    context_len = CONTEXT_LEN if context_len is None else context_len
    embed_size = EMBED_SIZE if embed_size is None else embed_size
    num_layers = NUM_LAYERS if num_layers is None else num_layers
    num_heads = NUM_HEADS if num_heads is None else num_heads
    hidden_size = HIDDEN_SIZE if hidden_size is None else hidden_size
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    lines = [
        "Architecture:",
        f"  context_len={context_len}",
        f"  embed_size={embed_size}",
        f"  num_layers={num_layers}",
        f"  num_heads={num_heads}",
        f"  hidden_size={hidden_size}",
        "Training:",
        f"  batch_size={batch_size}",
        f"  learning_rate={LEARNING_RATE}",
        f"  warmup_pct={WARMUP_PCT}",
        "Scheduling:",
        f"  log_interval_secs={LOG_INTERVAL_SECS}",
        f"  warmup_window_secs={WARMUP_WINDOW_SECS}",
        f"  plot_warmup_secs={PLOT_WARMUP_SECS}",
        f"  plot_interval_secs={PLOT_INTERVAL_SECS}",
        f"  checkpoint_warmup_secs={CHECKPOINT_WARMUP_SECS}",
        f"  checkpoint_interval_secs={CHECKPOINT_INTERVAL_SECS}",
    ]
    if model is not None:
        lines.insert(6, f"  param_count={_format_param_count(_model_param_count(model))}")
        lines.insert(7, f"  quantization={_model_quantization(model)}")
    print("\n".join(lines))


def print_chat_hyperparams(context_len, max_new_tokens, temperature, top_k, model=None):
    """Print the inference-related hyperparameters."""
    lines = [
        "Architecture:",
        f"  context_len={context_len}",
        "Inference:",
        f"  max_new_tokens={max_new_tokens}",
        f"  temperature={temperature}",
        f"  top_k={top_k}",
    ]
    if model is not None:
        lines.insert(2, f"  param_count={_format_param_count(_model_param_count(model))}")
        lines.insert(3, f"  quantization={_model_quantization(model)}")
    print("\n".join(lines))
