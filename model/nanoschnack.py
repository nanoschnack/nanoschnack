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
from pickletools import optimize

import torch

torch.backends.mps.is_available()
torch.backends.mps.is_built()

# %% [markdown]
# ## Trying out MPS

# %%
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


# %% [markdown]
# ## Loading a tokenizer with Hugging Face's tokenizer library
#
# - Compare: https://github.com/huggingface/tokenizers
# - Tiktokenizer: https://tiktokenizer.vercel.app/?model=gpt2

# %%
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

tokenizer_path = hf_hub_download(repo_id="openai-community/gpt2", filename="tokenizer.json")


# %% [markdown]
# ### Testing the tokenizer

# %%
tokenizer = Tokenizer.from_file(tokenizer_path)
print(tokenizer.encode("Hello, World!").ids)

# %% [markdown]
# ## Instantiating the NanoSchnack model

# %%
from gpt import GPT

model = GPT(vocab_size=tokenizer.get_vocab_size()).to(device).train()

# %% [markdown]
# ## Load the Training Data

# %%
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load dataset in streaming mode (does not load everything into memory at once)
# Note(sttts): I am using https://huggingface.co/datasets/pdelobelle/fineweb-german-edu-mt.
raw_ds = load_dataset(
    "parquet",
    data_files={"train": "../data/*.parquet"},
    split="train",
    streaming=True,
)

# Shuffle the dataset with a buffer for approximate shuffling
shuffled = raw_ds.shuffle(buffer_size=10_000, seed=42) # lazy shuffle (approximate) with a buffer

# Enable truncation and padding
tokenizer.enable_truncation(max_length=128)
tokenizer.enable_padding(length=128, pad_id=0, pad_token="[PAD]")

# Wrap Hugging Face tokenizer for batch processing
def tokenizer_batch(batch):
    token_batch = tokenizer.encode_batch(batch["result"])
    return {
        "input_ids": [e.ids for e in token_batch],
        "attention_mask": [e.attention_mask for e in token_batch],
    }
dataset = shuffled.map(tokenizer_batch, batched=True)
dataset = dataset.with_format(type="torch")

# Tokenize the dataset
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# %% [markdown]
# ## Run the Training

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10_000)
lossFn = torch.nn.CrossEntropyLoss()

steps_per_epoch = 10
for epoch in range(10):
    for step, batch in enumerate(loader):
        if step >= steps_per_epoch:
            break

        input_ids = batch["input_ids"].to(device)

        # Next-token prediction
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        loss = lossFn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()

    lossPerBit = loss.item() / (16 * 4) / torch.log(torch.tensor(2.0))
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Loss per bit: {lossPerBit:.6f}")
