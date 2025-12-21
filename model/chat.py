import argparse

import torch

try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths

from config import (
    CONTEXT_LEN,
    EMBED_SIZE,
    HIDDEN_SIZE,
    MAX_NEW_TOKENS,
    NUM_HEADS,
    NUM_LAYERS,
    TEMPERATURE,
    TOP_K,
    print_chat_hyperparams,
)
from device import device_info, pick_device, print_device_info
from gpt import GPT
from tokenizer import load_tokenizer


def load_model(checkpoint_path, vocab_size, device):
    # Construct the model and optionally load weights from checkpoint.
    model_config = {
        "context_len": CONTEXT_LEN,
        "embed_size": EMBED_SIZE,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "hidden_size": HIDDEN_SIZE,
    }
    state_dict = None

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
            if "config" in ckpt:
                model_config.update(ckpt["config"])
        else:
            state_dict = ckpt

    if state_dict is not None:
        if "position_embedding.weight" in state_dict:
            model_config["context_len"] = state_dict["position_embedding.weight"].shape[0]
        if "token_embedding.weight" in state_dict:
            vocab_size = state_dict["token_embedding.weight"].shape[0]

    model = GPT(
        vocab_size=vocab_size,
        embed_size=model_config["embed_size"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["num_heads"],
        hidden_size=model_config["hidden_size"],
        context_len=model_config["context_len"],
    ).to(device).eval()

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model, model_config["context_len"]


def sample_next_token(logits, temperature, top_k):
    # Sample a token using temperature and optional top-k truncation.
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    logits = logits / temperature
    if top_k and top_k > 0:
        values, indices = torch.topk(logits, min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, num_samples=1)
        return int(indices[choice].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_reply_stream(model, tokenizer, prompt, context_len, max_new_tokens, temperature, top_k, device):
    # Stream tokens from autoregressive decoding for a single reply.
    prompt_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([prompt_ids[-context_len:]], device=device, dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :].squeeze(0)
            next_id = sample_next_token(logits, temperature, top_k)
            yield tokenizer.decode([next_id])
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
            if input_ids.size(1) > context_len:
                input_ids = input_ids[:, -context_len:]


def run_repl(model, tokenizer, context_len, max_new_tokens, temperature, top_k, device):
    # Enable basic readline navigation if available.
    try:
        import readline  # noqa: F401
    except Exception:
        pass

    history = ""
    print("Type '/help' for commands.")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text == "/help":
            print("Commands: /help, /quit, /exit, /reset, /temp <value>, /topk <value>")
            continue
        if user_text.startswith("/temp"):
            parts = user_text.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Temperature: {temperature}")
            else:
                try:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Temperature must be a number.")
            continue
        if user_text.startswith("/topk"):
            parts = user_text.split(maxsplit=1)
            if len(parts) != 2:
                print(f"Top-k: {top_k}")
            else:
                try:
                    top_k = int(parts[1])
                    if top_k < 0:
                        raise ValueError
                    print(f"Top-k set to {top_k}")
                except ValueError:
                    print("Top-k must be a non-negative integer.")
            continue
        if user_text in {"/quit", "/exit"}:
            break
        if user_text == "/reset":
            history = ""
            print("History cleared.")
            continue

        history += f"User: {user_text}\nAssistant:"
        print("bot> ", end="", flush=True)
        reply_parts = []
        try:
            for token in generate_reply_stream(
                model,
                tokenizer,
                history,
                context_len=context_len,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device,
            ):
                print(token, end="", flush=True)
                reply_parts.append(token)
        except KeyboardInterrupt:
            print()
        reply = "".join(reply_parts).strip()
        if reply:
            history += f" {reply}\n"
        else:
            history += "\n"
        if reply_parts:
            print()


def main():
    parser = argparse.ArgumentParser(description="Chat with the NanoSchnack model.")
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint (.pt).")
    parser.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()

    # Resolve default checkpoint location and load components.
    model_dir, _, checkpoint_dir = setup_paths()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = checkpoint_dir / "latest.pt"
        if not checkpoint_path.exists():
            checkpoint_path = None

    device = pick_device()
    info = device_info(device)
    print_device_info(info)
    tokenizer = load_tokenizer()

    model, model_context_len = load_model(checkpoint_path, tokenizer.get_vocab_size(), device)

    print_chat_hyperparams(
        model_context_len,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        model=model,
    )

    # Validate that the requested context length fits the trained model.
    if args.context_len > model_context_len:
        raise ValueError(
            f"--context-len ({args.context_len}) exceeds trained context length ({model_context_len})."
        )

    run_repl(
        model,
        tokenizer,
        context_len=args.context_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )


if __name__ == "__main__":
    main()
