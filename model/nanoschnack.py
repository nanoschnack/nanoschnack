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

# add special tokens
tokenizer.add_special_tokens(["[PAD]"])
pad_id = tokenizer.token_to_id("[PAD]")

context_len = 256
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

# do or not do chunking of the input text, instead of truncating.
if False:
    max_len = context_len
    stride = context_len/4  # overlap; set to 0 for no overlap

    tokenizer.disable_truncation()
    tokenizer.disable_padding()

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
        attention_mask = []
        for text in batch["result"]:
            ids = tokenizer.encode(text).ids
            for chunk in chunk_ids(ids, max_len=max_len,
                                   stride=stride):
                input_ids.append(chunk)
                attention_mask.append([1 if t != pad_id else 0 for t
                                       in chunk])
        return {"input_ids": input_ids, "attention_mask": attention_mask}
else:
    # Enable truncation and padding
    tokenizer.enable_truncation(max_length=context_len)
    tokenizer.enable_padding(length=context_len, pad_id=pad_id, pad_token="[PAD]")

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
batch_size = 32
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

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

        # flatten the output and targets into lists for batch loss computation.
        loss = lossFn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        loss.backward()
        optimizer.step()
        scheduler.step()

    lossPerBit = loss.item() / (16 * 4) / torch.log(torch.tensor(2.0))
    print(f"Epoch {epoch+1}, Loss: {loss.item()}, Loss per bit: {lossPerBit:.6f}")
