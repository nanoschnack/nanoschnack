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
    WARMUP_WINDOW_SECS,
    print_training_hyperparams,
)

# add special tokens
tokenizer.add_special_tokens(["[PAD]"])
pad_id = tokenizer.token_to_id("[PAD]")

context_len = CONTEXT_LEN
print_training_hyperparams()
model = GPT(
    vocab_size=tokenizer.get_vocab_size(),
    embed_size=EMBED_SIZE,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    hidden_size=HIDDEN_SIZE,
    context_len=CONTEXT_LEN,
).to(device).train()



# %% [markdown]
# ## Load the Training Data

# %%
# Resolve model paths so relative data/checkpoint locations are stable.
try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths
model_dir, data_dir, checkpoint_dir = setup_paths()

from datasets import load_dataset
from torch.utils.data import DataLoader
# Load dataset in streaming mode (does not load everything into memory at once)
# Note(sttts): I am using https://huggingface.co/datasets/pdelobelle/fineweb-german-edu-mt.
raw_ds = load_dataset(
    "parquet",
    data_files={"train": str(data_dir / "*.parquet")},
    split="train",
    streaming=True,
)

# Shuffle the dataset with a buffer for approximate shuffling
shuffled = raw_ds.shuffle(buffer_size=10_000, seed=42) # lazy shuffle (approximate) with a buffer

# do or not do chunking of the input text, instead of truncating.
if False:
    max_len = context_len
    stride = context_len/4  # overlap; set to 0 for no overlap

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
        for text in batch["result"]:
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
        token_batch = tokenizer.encode_batch(batch["result"])
        return {
            "input_ids": [e.ids for e in token_batch],
            "attention_mask": [e.attention_mask for e in token_batch], # marks real tokens (1) vs padding (0)
        }

# Shuffle deterministically (only way for streaming datasets)
dataset = shuffled.map(tokenizer_batch, batched=True)

# Set the dataset format to PyTorch tensors
dataset = dataset.with_format(type="torch")

# Tokenize the dataset
batch_size = BATCH_SIZE
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# %% [markdown]
# ## Run the Training

# %%
from plot import ascii_loss_plot
from progress import ProgressLogger
from checkpointer import Checkpointer
import torch
import time

# Set up optimizer, learning-rate scheduler, and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10_000)
lossFn = torch.nn.CrossEntropyLoss()

# The checkpointer will save and load model/optimizer/scheduler states to/from disk.
checkpointer = Checkpointer(checkpoint_dir, model, optimizer, scheduler, device=device)
resume_epoch, resume_step, global_step = checkpointer.load_latest()

# Initialize the progress logger to display training progress and loss
progress = ProgressLogger(
    ascii_loss_plot,
    start_global_step=global_step,
    log_interval=LOG_INTERVAL_SECS,
    warmup_plot_interval=PLOT_WARMUP_SECS,
    plot_interval=PLOT_INTERVAL_SECS,
    warmup_window_secs=WARMUP_WINDOW_SECS,
)
last_ckpt_time = time.time()

epochs = 1 # epochs between 1 and 3 are usually sufficient for good results, rather 1 than 3.
for epoch in range(resume_epoch, epochs):
    for step, batch in enumerate(loader):
        if epoch == resume_epoch and step < resume_step:
            continue
        # Get the input IDs and attention mask, and move them to the GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Next-token prediction
        inputs = input_ids[:, :-1] # everything from the first token except the last
        targets = input_ids[:, 1:] # everything from the second token onward

        # Clear accumulated gradients from the previous step (which torch does automatically otherwise)
        optimizer.zero_grad()

        # Forward pass
        logits = model(inputs, attention_mask=attention_mask[:, :-1])

        # Compute (average) loss of the predicted next tokens and apply backpropagation.
        # reshape to (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
        loss = lossFn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()

        # Update weights, then advance the learning-rate schedule.
        optimizer.step()
        scheduler.step()

        # Log progress and plot loss history
        now = time.time()
        ckpt_interval = CHECKPOINT_WARMUP_SECS if (now - last_ckpt_time) < WARMUP_WINDOW_SECS else CHECKPOINT_INTERVAL_SECS
        if now - last_ckpt_time >= ckpt_interval:
            checkpointer.save_latest(epoch, step, progress.global_step)
            last_ckpt_time = now
        progress.tick(loss.item(), input_ids.size(0), epoch, step)

