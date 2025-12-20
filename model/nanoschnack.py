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
torch.backends.mps.is_available()
torch.backends.mps.is_built()

# %% [markdown]
# ## Trying out MPS

# %%
device = torch.device("mps")
x = torch.randn(1024,1024, device=device)

# %% [markdown]
# ## Loading a tokenizer with Hugging Face's tokenizer library
#
# Compare: https://github.com/huggingface/tokenizers

# %%
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

tokenizer_path = hf_hub_download(repo_id="openai-community/gpt2", filename="tokenizer.json")


# %% [markdown]
# ## Testing the tokenizer

# %%
tokenizer = Tokenizer.from_file(tokenizer_path)
output = tokenizer.encode("Hello, World!")
print(output.ids)
