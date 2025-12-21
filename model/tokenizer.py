from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


def load_tokenizer():
    # Load the tokenizer and align special tokens with training.
    tokenizer_path = hf_hub_download(repo_id="openai-community/gpt2", filename="tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Keep the vocab aligned with training if a pad token was added.
    if tokenizer.token_to_id("[PAD]") is None:
        tokenizer.add_special_tokens(["[PAD]"])

    return tokenizer
