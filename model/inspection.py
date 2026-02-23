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
# # NanoSchnack Model Inspection
#
# Explore learned token embeddings from a saved checkpoint:
# - Tokenize words and inspect their token IDs
# - Retrieve and compare raw embedding vectors
# - Compute cosine similarity between words
# - Do word-vector arithmetic (e.g. Heidelberg − Stadt + Essen ≈ ?)

# %% Setup
import sys
import torch
import torch.nn.functional as F

# Add the model directory to the path for local imports.
try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths

model_dir, data_dir, checkpoint_dir = setup_paths()

# %% [markdown]
# ## Load checkpoint config, tokenizer, and model

# %%
from checkpointer import load_checkpoint_config, select_state_dict, normalize_state_dict, load_model_state_dict
from tokenizer import load_tokenizer, print_vocab_alignment
from gpt import GPT
import config

checkpoint_path = checkpoint_dir / "latest.pt"
ckpt = load_checkpoint_config(checkpoint_path)

tokenizer = load_tokenizer()
print_vocab_alignment(tokenizer)

model = GPT(
    vocab_size=tokenizer.get_vocab_size(),
    embed_size=config.EMBED_SIZE,
    num_layers=config.NUM_LAYERS,
    num_heads=config.NUM_HEADS,
    hidden_size=config.HIDDEN_SIZE,
    context_len=config.CONTEXT_LEN,
    pos_embed_type=config.POS_EMBED_TYPE,
    rope_base=config.ROPE_BASE,
)

# Load model weights from the checkpoint.
state_dict = select_state_dict(ckpt)
state_dict = normalize_state_dict(state_dict)
load_model_state_dict(model, state_dict)
model.eval()

param_count = sum(p.numel() for p in model.parameters())
print(f"Model loaded: {param_count:,} parameters, vocab_size={tokenizer.get_vocab_size()}")

# %% [markdown]
# ## Word → Tokens → Embeddings
#
# Each word is split into one or more subword tokens.  The embedding for a
# multi-token word is the mean of its token embeddings, identical to the
# approach used in many static-embedding baselines.

# %%
# Embedding matrix shared between input and output projection.
E = model.tok.weight.detach()  # shape: (vocab_size, embed_size)

def word_tokens(word: str) -> list[int]:
    """Return token IDs for *word* with a leading space (in-context BPE form)."""
    return tokenizer.encode(" " + word).ids

def word_embedding(word: str) -> torch.Tensor:
    """Return the mean token embedding for *word*, L2-normalised."""
    ids = word_tokens(word)
    vecs = E[ids]           # (n_tokens, embed_size)
    mean = vecs.mean(dim=0) # (embed_size,)
    return F.normalize(mean, dim=0)

def show_tokens(word: str) -> None:
    ids = word_tokens(word)
    decoded = [tokenizer.decode([i]) for i in ids]
    print(f"  {word!r:20s} → ids={ids}  tokens={decoded}")

print("Tokenisation:")
for w in [
    # cities
    "Berlin", "Dresden", "Wiesbaden", "Paris",
    # countries / states
    "Deutschland", "Frankreich", "Bayern", "Sachsen", "Hessen",
    # city/country descriptors
    "Stadt", "Hauptstadt",
    # family (gender analogy)
    "Vater", "Mutter", "Mann", "Frau",
    # animals (plural analogy)
    "Hund", "Hunde", "Katze", "Katzen",
    # professions (gender analogy)
    "Lehrer", "Lehrerin",
    # drinks (cosine similarity)
    "Bier", "Wein",
    # transport (cosine similarity)
    "Auto", "Zug",
]:
    show_tokens(w)

print()
print("Tokenizer struggles:")
for w in [
    # umlauts get shredded into UTF-8 byte pieces
    "München", "Köln", "König", "Österreich", "Ärztin",
    # long compounds
    "Donaudampfschifffahrtsgesellschaft", "Kraftfahrzeughaftpflichtversicherung",
    # food with rare endings
    "Bratwurst", "Saumagen", "Dampfnudeln",
]:
    show_tokens(w)

# %% [markdown]
# ## Cosine Similarity Between Words

# %%
def cosine_sim(a: str, b: str) -> float:
    """Cosine similarity between the embeddings of words *a* and *b*."""
    return float(word_embedding(a) @ word_embedding(b))

pairs = [
    # related
    ("Berlin",      "Hauptstadt"),
    ("Deutschland", "Frankreich"),
    ("Vater",       "Mutter"),
    ("Hund",        "Katze"),
    ("Bier",        "Wein"),
    ("Auto",        "Zug"),
    # unrelated
    ("Hund",        "Zug"),
    ("Bier",        "Sachsen"),
    ("Vater",       "Wiesbaden"),
    ("Lehrer",      "Bier"),
]

print("Cosine similarities:")
for a, b in pairs:
    sim = cosine_sim(a, b)
    bar = "█" * int(abs(sim) * 30)
    print(f"  {a:15s} ↔ {b:15s}  {sim:+.4f}  {bar}")

# %% [markdown]
# ## Word-Vector Arithmetic
#
# The canonical analogy: **König − Mann + Frau ≈ Königin**
#
# We compute the query vector, then find the *k* nearest neighbours in the
# full embedding matrix by cosine similarity, skipping the query words
# themselves so the result is informative.
#
# > **Note:** Analogy arithmetic requires well-structured embedding geometry,
# > which only emerges after extensive training.  Re-run this cell periodically
# > to watch the results improve as the model trains.

# %%
def nearest_neighbours(query_vec: torch.Tensor, k: int = 10, exclude: list[str] | None = None) -> list[tuple[str, float]]:
    """Return the *k* tokens whose embeddings are closest to *query_vec*."""
    exclude_ids: set[int] = set()
    for w in (exclude or []):
        exclude_ids.update(word_tokens(w))

    # Normalise the embedding matrix row-wise once per call.
    E_norm = F.normalize(E, dim=1)           # (vocab_size, embed_size)
    q_norm = F.normalize(query_vec, dim=0)   # (embed_size,)
    sims = E_norm @ q_norm                   # (vocab_size,)

    # Mask excluded tokens so they can't win.
    for idx in exclude_ids:
        sims[idx] = -1.0

    top_ids = sims.topk(k).indices.tolist()
    return [(tokenizer.decode([i]), float(sims[i])) for i in top_ids]


def word_math(positive: list[str], negative: list[str], k: int = 10) -> list[tuple[str, float]]:
    """Analogy via vector arithmetic: sum(positive) − sum(negative)."""
    vec = sum(word_embedding(w) for w in positive) - sum(word_embedding(w) for w in negative)
    exclude = positive + negative
    return nearest_neighbours(vec, k=k, exclude=exclude)


def show_analogy_tokens(positive: list[str], negative: list[str]) -> None:
    """Print the token breakdown for every word involved in an analogy."""
    for w in positive:
        ids = word_tokens(w)
        decoded = [tokenizer.decode([i]) for i in ids]
        print(f"  +{w!r:19s} → ids={ids}  tokens={decoded}")
    for w in negative:
        ids = word_tokens(w)
        decoded = [tokenizer.decode([i]) for i in ids]
        print(f"  -{w!r:19s} → ids={ids}  tokens={decoded}")


# Vater − Mann + Frau ≈ ?  (expect: Mutter)
print("Vater − Mann + Frau  →  Mutter?")
show_analogy_tokens(positive=["Vater", "Frau"], negative=["Mann"])
print("results:")
for token, sim in word_math(positive=["Vater", "Frau"], negative=["Mann"]):
    print(f"  {token!r:20s}  {sim:+.4f}")

# %% [markdown]
# ## More Analogies
#
# Each row encodes one clear relationship **A : B = C : D**, expressed as
# **A − B + D ≈ C**.  The expected answer is noted after the arrow.

# %%
analogies = [
    # (positive, negative, label)                                                      relationship
    (["Vater",   "Frau"],       ["Mann"],        "Vater − Mann + Frau  →  Mutter?"),   # family gender
    (["Berlin",  "Frankreich"], ["Deutschland"], "Berlin − Deutschland + Frankreich  →  Paris?"),  # capital city
    (["Dresden", "Hessen"],     ["Sachsen"],     "Dresden − Sachsen + Hessen  →  Wiesbaden?"),     # state capital
    (["Hunde",   "Katze"],      ["Hund"],        "Hunde − Hund + Katze  →  Katzen?"),             # plural
    (["Lehrer",  "Frau"],       ["Mann"],        "Lehrer − Mann + Frau  →  Lehrerin?"),           # profession gender
]

for pos, neg, label in analogies:
    print(f"\n{label}")
    show_analogy_tokens(positive=pos, negative=neg)
    print("results:")
    for token, sim in word_math(positive=pos, negative=neg, k=5):
        print(f"  {token!r:20s}  {sim:+.4f}")
