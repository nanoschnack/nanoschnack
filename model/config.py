"""Shared hyperparameters for training and inference."""

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
