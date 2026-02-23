# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # GPT-2 Embedding Inspection (Reference)
#
# Same inspection as `inspection.ipynb`, but using the pretrained English GPT-2
# (117 M parameters, ~10 B tokens of WebText) as a reference point.
#
# Key differences from NanoSchnack:
# - English vocabulary, whole-word BPE tokenization
# - Fully trained — analogy arithmetic works reliably
# - Raw `wte` embedding matrix (same architectural position as `tok.weight`)

# %%
import torch
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
model.eval()

vocab_size, embed_size = model.wte.weight.shape
param_count = sum(p.numel() for p in model.parameters())
print(f"GPT-2 loaded: {param_count:,} parameters")
print(f"vocab_size={vocab_size}  embed_size={embed_size}")

# %% [markdown]
# ## Word → Tokens → Embeddings
#
# GPT-2 uses byte-pair encoding (BPE).  Common words appearing mid-sentence are
# encoded **with a leading space** (e.g. `" king"` → one token).  We add a space
# prefix so standalone words get their in-context representation.

# %%
# Embedding matrix (vocab_size, embed_size) — equivalent to tok.weight in NanoSchnack.
E = model.wte.weight.detach()

def word_tokens(word: str) -> list[int]:
    """Return token IDs; prepend a space so mid-sentence BPE applies."""
    return tokenizer.encode(" " + word)

def word_embedding(word: str) -> torch.Tensor:
    """Mean L2-normalised token embedding for *word*."""
    ids = word_tokens(word)
    return F.normalize(E[ids].mean(dim=0), dim=0)

def show_tokens(word: str) -> None:
    ids = word_tokens(word)
    decoded = [tokenizer.decode([i]) for i in ids]
    print(f"  {word!r:15s} → ids={ids}  tokens={decoded}")

print("Tokenisation:")
for w in ["king", "queen", "man", "woman", "Paris", "Berlin", "France", "Germany", "dogs", "cats"]:
    show_tokens(w)


# %% [markdown]
# ## Cosine Similarity Between Words

# %%
def cosine_sim(a: str, b: str) -> float:
    return float(word_embedding(a) @ word_embedding(b))

pairs = [
    ("king",   "queen"),
    ("king",   "man"),
    ("Paris",  "Berlin"),
    ("Paris",  "London"),
    ("dog",    "cat"),
    ("dog",    "mathematics"),
    ("walk",   "run"),
    ("walk",   "democracy"),
]

print("Cosine similarities:")
for a, b in pairs:
    sim = cosine_sim(a, b)
    bar = "█" * int(abs(sim) * 40)
    print(f"  {a:12s} ↔ {b:12s}  {sim:+.4f}  {bar}")


# %% [markdown]
# ## Word-Vector Arithmetic
#
# The canonical analogy: **king − man + woman ≈ queen**
#
# GPT-2 is fully trained, so this works.  Compare the results with
# `inspection.ipynb` to gauge NanoSchnack's progress.

# %%
def nearest_neighbours(query_vec: torch.Tensor, k: int = 10, exclude: list[str] | None = None) -> list[tuple[str, float]]:
    """Return the *k* tokens whose embeddings are closest to *query_vec*."""
    exclude_ids: set[int] = set()
    for w in (exclude or []):
        exclude_ids.update(word_tokens(w))

    # Normalise the embedding matrix row-wise once per call.
    E_norm = F.normalize(E, dim=1)
    q_norm = F.normalize(query_vec, dim=0)
    sims = E_norm @ q_norm

    # Mask excluded tokens so they can't win.
    for idx in exclude_ids:
        sims[idx] = -1.0

    top_ids = sims.topk(k).indices.tolist()
    return [(tokenizer.decode([i]), float(sims[i])) for i in top_ids]


def word_math(positive: list[str], negative: list[str], k: int = 10) -> list[tuple[str, float]]:
    """Analogy via vector arithmetic: sum(positive) − sum(negative)."""
    vec = sum(word_embedding(w) for w in positive) - sum(word_embedding(w) for w in negative)
    return nearest_neighbours(vec, k=k, exclude=positive + negative)


# king − man + woman ≈ ?  (expect: queen)
print("king − man + woman  →  queen?")
for token, sim in word_math(positive=["king", "woman"], negative=["man"]):
    print(f"  {token!r:20s}  {sim:+.4f}")

# %% [markdown]
# ## More Analogies
#
# Each row encodes one clear relationship **A : B = C : D**, expressed as
# **A − B + D ≈ C**.  The expected answer is noted after the arrow.

# %%
analogies = [
    # (positive, negative, label)                                                    relationship
    (["king",   "woman"],     ["man"],       "king − man + woman  →  queen?"),        # gender
    (["uncle",  "woman"],     ["man"],       "uncle − man + woman  →  aunt?"),         # family gender
    (["Paris",  "Germany"],   ["France"],    "Paris − France + Germany  →  Berlin?"),  # capital city
    (["Berlin", "France"],    ["Germany"],   "Berlin − Germany + France  →  Paris?"),  # capital city (reversed)
    (["dogs",   "cat"],       ["dog"],       "dogs − dog + cat  →  cats?"),            # plural
]

for pos, neg, label in analogies:
    print(f"\n{label}")
    for token, sim in word_math(positive=pos, negative=neg, k=5):
        print(f"  {token!r:20s}  {sim:+.4f}")
