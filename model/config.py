"""Shared hyperparameters for training and inference."""

import torch

###
### Model architecture
###

# Maximum sequence length used to size positional embeddings.
# Keep this aligned between training and inference.
CONTEXT_LEN = 256

# Embedding dimensionality for token and position representations.
# Larger values increase model capacity and compute cost.
EMBED_SIZE = 512

# Number of Transformer encoder layers in the model.
# Higher values deepen the network and increase training time.
NUM_LAYERS = 3

# Number of attention heads per Transformer layer.
# Must divide EMBED_SIZE evenly.
NUM_HEADS = 8

# Feed-forward hidden size inside each Transformer layer.
# Often 4x EMBED_SIZE for Transformer blocks.
HIDDEN_SIZE = 2048

###
### Training defaults
###

# Default batch size for training loops.
# Adjust based on device memory.
BATCH_SIZE = 32

# Default learning rate for the optimizer.
# Tune alongside batch size and scheduler.
LEARNING_RATE = 1e-4

###
### Inference defaults
###

# Default maximum number of tokens to generate per reply.
# Increase for longer responses at higher compute cost.
MAX_NEW_TOKENS = 128

# Sampling temperature for inference (0 disables sampling).
# Lower values are more deterministic.
TEMPERATURE = 0.8

# Top-k cutoff for sampling (0 disables top-k filtering).
# Use 0 to sample from the full distribution.
TOP_K = 50

###
### Logging and checkpoint cadence
###

# Checkpoint cadence in seconds after the warmup window.
# Used after the initial warmup period.
CHECKPOINT_INTERVAL_SECS = 600

# Checkpoint cadence in seconds during the first 10 minutes.
# Frequent saves help capture early training progress.
CHECKPOINT_WARMUP_SECS = 60

# Warmup window length in seconds before switching to standard intervals.
# Shared by plotting and checkpoint scheduling.
WARMUP_WINDOW_SECS = 600

# Plot cadence in seconds after the warmup window.
# Controls how often loss charts are printed.
PLOT_INTERVAL_SECS = 600

# Plot cadence in seconds during the first 10 minutes.
# Higher frequency helps during the startup phase.
PLOT_WARMUP_SECS = 60

# Log cadence in seconds for progress updates.
# Impacts console verbosity and throughput reporting.
LOG_INTERVAL_SECS = 10


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


def print_training_hyperparams(model=None):
    """Print the training-related hyperparameters."""
    lines = [
        "Architecture:",
        f"  context_len={CONTEXT_LEN}",
        f"  embed_size={EMBED_SIZE}",
        f"  num_layers={NUM_LAYERS}",
        f"  num_heads={NUM_HEADS}",
        f"  hidden_size={HIDDEN_SIZE}",
        "Training:",
        f"  batch_size={BATCH_SIZE}",
        f"  learning_rate={LEARNING_RATE}",
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
