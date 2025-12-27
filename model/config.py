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

def _env_str(name, default):
    return _env_override(name, str, default)


# Maximum sequence length used to size positional embeddings.
# Keep this aligned between training and inference.
CONTEXT_LEN = _env_int("CONTEXT_LEN", 1024)

# Vocabulary size for token embeddings and output projection.
# Use 0 to auto-align the tokenizer size to a nearby power-of-two multiple.
VOCAB_SIZE = _env_int("VOCAB_SIZE", 0)

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
HIDDEN_SIZE = _env_int("HIDDEN_SIZE", 4*EMBED_SIZE)

###
### Training defaults
###

# Default batch size for training loops.
# Adjust based on device memory.
BATCH_SIZE = _env_int("BATCH_SIZE", 32)

# Default macro batch size for gradient accumulation.
# Must be divisible by BATCH_SIZE.
MACRO_BATCH_SIZE = _env_int("MACRO_BATCH_SIZE", 512)

# Default learning rate for the optimizer.
# Source: GPT-3 paper Table 2 (GPT-3 Small 125M uses 6.0e-4),
# https://arxiv.org/src/2005.14165
LEARNING_RATE = _env_float("LEARNING_RATE", 6e-4)

# Warmup fraction of total training steps for LR ramp-up.
# Use values between 0.01 and 0.05 for small warmups.
WARMUP_PCT = _env_float("WARMUP_PCT", 0.03)

# Maximum training tokens multiplier relative to parameter count (0 = unlimited).
MAX_TRAINING_FACTOR = _env_int("MAX_TRAINING_FACTOR", 20)

# Shuffle buffer size for streaming datasets.
SHUFFLE_BUFFER = _env_int("SHUFFLE_BUFFER", 10_000)

# Batch size for dataset packing during tokenization.
PACK_BATCH_SIZE = _env_int("PACK_BATCH_SIZE", 1000)

###
### Dataset defaults
###

# Comma-separated dataset specs: hf:<repo_id>[:split][:text_key], hf:<repo_id>:<config>:<split>[:text_key], txt:<path>[:text_key]
DATASET_SPECS = _env_str(
    "DATASET_SPECS",
    "hf:coral-nlp/german-commons:web:onemillionposts:text,"
    "hf:coral-nlp/german-commons:web:wikipedia:text,"
    "hf:coral-nlp/german-commons:web:youtubecommons:text,"
    "hf:coral-nlp/german-commons:cultural:wikivoyage:text,"
    "hf:coral-nlp/german-commons:cultural:wikiquote:text",
)

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

# Prompt used for sample completions appended to loss plots.
PLOT_COMPLETION_PROMPT = _env_str("PLOT_COMPLETION_PROMPT", "Die Hauptstadt von Deutschland")

# Number of tokens to generate for plot completions.
PLOT_COMPLETION_TOKENS = _env_int("PLOT_COMPLETION_TOKENS", 128)



def snapshot():
    return {
        "CONTEXT_LEN": CONTEXT_LEN,
        "VOCAB_SIZE": VOCAB_SIZE,
        "EMBED_SIZE": EMBED_SIZE,
        "NUM_LAYERS": NUM_LAYERS,
        "NUM_HEADS": NUM_HEADS,
        "HIDDEN_SIZE": HIDDEN_SIZE,
    }


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


def _architecture_lines():
    lines = [
        "Architecture:",
        f"  context_len={CONTEXT_LEN}",
        f"  vocab_size={VOCAB_SIZE}",
        f"  embed_size={EMBED_SIZE}",
        f"  num_layers={NUM_LAYERS}",
        f"  num_heads={NUM_HEADS}",
        f"  hidden_size={HIDDEN_SIZE}",
    ]
    return lines


def model_info(model):
    """Return model metadata for hyperparameter printouts."""
    return _model_param_count(model), _model_quantization(model)


def print_training_hyperparams(param_count=None, quantization=None):
    """Print the training-related hyperparameters."""
    lines = _architecture_lines() + [
        "Training:",
        f"  batch_size={BATCH_SIZE}",
        f"  macro_batch_size={MACRO_BATCH_SIZE}",
        f"  learning_rate={LEARNING_RATE}",
        f"  warmup_pct={WARMUP_PCT}",
        f"  max_training_factor={MAX_TRAINING_FACTOR}",
        f"  shuffle_buffer={SHUFFLE_BUFFER}",
        f"  dataset_specs={DATASET_SPECS}",
    ]
    if param_count is not None or quantization is not None:
        lines += [
            "Model:",
            f"  param_count={_format_param_count(param_count)}",
            f"  quantization={quantization}",
        ]
    lines += [
        "Scheduling:",
        f"  warmup_window_secs={WARMUP_WINDOW_SECS}",
        f"  plot_warmup_secs={PLOT_WARMUP_SECS}",
        f"  plot_interval_secs={PLOT_INTERVAL_SECS}",
        f"  plot_completion_tokens={PLOT_COMPLETION_TOKENS}",
        f"  plot_completion_prompt={PLOT_COMPLETION_PROMPT}",
        f"  checkpoint_warmup_secs={CHECKPOINT_WARMUP_SECS}",
        f"  checkpoint_interval_secs={CHECKPOINT_INTERVAL_SECS}",
    ]
    print("\n".join(lines))


def align_micro_batch_size(micro_batch_size, macro_batch_size):
    """Return the largest divisor of macro_batch_size <= micro_batch_size."""
    if micro_batch_size <= 0 or macro_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    for candidate in range(micro_batch_size, 0, -1):
        if macro_batch_size % candidate == 0:
            return candidate
    return 1


def print_chat_hyperparams(param_count=None, quantization=None):
    """Print the inference-related hyperparameters."""
    lines = _architecture_lines() + [
        "Inference:",
        f"  max_new_tokens={MAX_NEW_TOKENS}",
        f"  temperature={TEMPERATURE}",
        f"  top_k={TOP_K}",
    ]
    if param_count is not None or quantization is not None:
        lines += [
            "Model:",
            f"  param_count={_format_param_count(param_count)}",
            f"  quantization={quantization}",
        ]
    print("\n".join(lines))
