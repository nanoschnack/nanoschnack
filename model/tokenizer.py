from huggingface_hub import hf_hub_download
import importlib.util
from pathlib import Path
from tokenizers import Tokenizer
import math
# Load the local config module directly to avoid cross-project name collisions.
_CONFIG_PATH = Path(__file__).resolve().parent / "config.py"
_spec = importlib.util.spec_from_file_location("nanoschnack_config", _CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load config from {_CONFIG_PATH}")
_local_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_local_config)


def _alignment_report(base_size, aligned_size):
    if base_size <= 0:
        return {
            "base_size": base_size,
            "aligned_size": aligned_size,
            "power": 1,
            "increase_pct": 0.0,
        }
    increase_pct = (aligned_size - base_size) / base_size * 100.0
    power = aligned_size & -aligned_size
    return {
        "base_size": base_size,
        "aligned_size": aligned_size,
        "power": power,
        "increase_pct": increase_pct,
    }


def _aligned_vocab_size(base_size, max_increase_ratio=0.02):
    limit = int(math.floor(base_size * (1.0 + max_increase_ratio)))
    best = None
    power = 1
    while power <= base_size:
        aligned = int(math.ceil(base_size / power) * power)
        if aligned <= limit:
            best = aligned
        power *= 2
    return best or base_size


def ensure_vocab_size(tokenizer, target_size):
    current_size = tokenizer.get_vocab_size()
    resolved_size = target_size
    if not resolved_size:
        resolved_size = _aligned_vocab_size(current_size)
        _local_config.VOCAB_SIZE = resolved_size
    if resolved_size == current_size:
        return current_size
    if resolved_size < current_size:
        raise ValueError(
            f"Target vocab size {resolved_size} is smaller than tokenizer size {current_size}."
        )
    needed = resolved_size - current_size
    extra_tokens = []
    counter = 0
    while len(extra_tokens) < needed:
        token = f"[EXTRA_{counter:06d}]"
        if tokenizer.token_to_id(token) is None:
            extra_tokens.append(token)
        counter += 1
    tokenizer.add_special_tokens(extra_tokens)
    new_size = tokenizer.get_vocab_size()
    if new_size < resolved_size:
        raise RuntimeError(f"Failed to reach vocab size {resolved_size}, got {new_size}.")
    return new_size


def print_vocab_alignment(tokenizer):
    # Report alignment diagnostics for tokenizer padding.
    alignment = getattr(tokenizer, "vocab_alignment", None)
    if not alignment:
        return
    print("Tokenizer:")
    print(f"  file={_local_config.TOKENIZER_FILENAME}")
    base_size = alignment["base_size"]
    print(f"  Tokenizer vocab size (base): {base_size}")
    print(
        "  Tokenizer vocab alignment: "
        f"base={alignment['base_size']} "
        f"aligned={alignment['aligned_size']} "
        f"power={alignment['power']} "
        f"(+{alignment['increase_pct']:.3f}%)"
    )


PAD_TOKEN = "<|PAD|>"
DATASET_EOS_TOKEN = "<|EOS|>"
BOS_TOKEN = "[BOS]"


def load_tokenizer():
    # Load the tokenizer and align special tokens with training.
    tokenizer_path = _resolve_tokenizer_path()
    if tokenizer_path is None:
        tokenizer_path = hf_hub_download(
            repo_id="openai-community/gpt2",
            filename="tokenizer.json",
        )
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Keep the vocab aligned with training if a pad token was added.
    if tokenizer.token_to_id(PAD_TOKEN) is None:
        tokenizer.add_special_tokens([PAD_TOKEN])
    if tokenizer.token_to_id(DATASET_EOS_TOKEN) is None:
        tokenizer.add_special_tokens([DATASET_EOS_TOKEN])

    # Align after adding special tokens so padding is counted in the base size.
    base_size = tokenizer.get_vocab_size()
    resolved_size = ensure_vocab_size(tokenizer, _local_config.VOCAB_SIZE)
    # Attach alignment metadata for training diagnostics.
    tokenizer.vocab_alignment = _alignment_report(base_size, resolved_size)
    return tokenizer


def _resolve_tokenizer_path():
    # Resolve TOKENIZER_JSON_PATH relative to the repo root when needed.
    tokenizer_path = getattr(_local_config, "TOKENIZER_JSON_PATH", "")
    if not tokenizer_path:
        tokenizer_dir = Path(__file__).resolve().parent.parent / "tokenizer"
        tokenizer_name = getattr(_local_config, "TOKENIZER_FILENAME", "tokenizer.json")
        candidate = (tokenizer_dir / tokenizer_name).resolve()
        if candidate.is_file():
            return str(candidate)
        fallback = (tokenizer_dir / "tokenizer.json").resolve()
        if fallback.is_file():
            return str(fallback)
        return str(candidate)
    candidate = Path(tokenizer_path)
    if not candidate.is_absolute():
        candidate = (Path(__file__).resolve().parent.parent / candidate).resolve()
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"Tokenizer JSON not found at {candidate}")
