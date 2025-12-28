import argparse

import torch

try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths

import config
from device import device_info, pick_device, print_device_info
from gpt import GPT
from checkpointer import (
    load_model_state_dict,
    normalize_state_dict,
    resize_vocab_state_dict,
    select_state_dict,
)
from tokenizer import DATASET_EOS_TOKEN, load_tokenizer


def load_model(checkpoint_path, vocab_size, device):
    # Construct the model and optionally load weights from checkpoint.
    state_dict = None
    ckpt_vocab_size = None

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            ckpt_vocab_size = ckpt.get("vocab_size")
        state_dict = select_state_dict(ckpt)
        if state_dict is not None:
            state_dict = normalize_state_dict(state_dict)

    model = GPT(
        vocab_size=vocab_size,
        embed_size=config.EMBED_SIZE,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        context_len=config.CONTEXT_LEN,
    ).to(device).eval()

    if state_dict is not None:
        try:
            resize_vocab_state_dict(model, state_dict, ckpt_vocab_size=ckpt_vocab_size)
            load_model_state_dict(model, state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to load checkpoint weights from {checkpoint_path}."
            ) from exc

    return model, config.CONTEXT_LEN


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

    use_chat_template = False
    show_tokens = False
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
            print("Commands: /help, /quit, /exit, /reset, /temp <value>, /topk <value>, /chat on|off, /debug on|off")
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
            print("History cleared.")
            continue
        if user_text.startswith("/chat"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                use_chat_template = not use_chat_template
            elif parts[1] in {"on", "off"}:
                use_chat_template = parts[1] == "on"
            else:
                print("Usage: /chat [on|off]")
                continue
            print(f"Chat template {'enabled' if use_chat_template else 'disabled'}.")
            continue
        if user_text.startswith("/debug"):
            parts = user_text.split(maxsplit=1)
            if len(parts) == 1:
                show_tokens = not show_tokens
            elif parts[1] in {"on", "off"}:
                show_tokens = parts[1] == "on"
            else:
                print("Usage: /debug [on|off]")
                continue
            print(f"Debug tokens {'enabled' if show_tokens else 'disabled'}.")
            continue

        if use_chat_template:
            prompt = f"User: {user_text}\nAssistant:"
        else:
            prompt = f"{DATASET_EOS_TOKEN}{user_text}"

        if show_tokens:
            print(f"tokens> {tokenizer.encode(user_text).ids}")
        print("bot> ", end="", flush=True)
        reply_parts = []
        try:
            for token in generate_reply_stream(
                model,
                tokenizer,
                prompt,
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
        if reply_parts:
            print()


def main():
    parser = argparse.ArgumentParser(description="Chat with the NanoSchnack model.")
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint (.pt).")
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=config.MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE)
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
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
    # Confirm base tokenizer size before alignment padding.
    alignment = getattr(tokenizer, "vocab_alignment", None)
    base_size = alignment["base_size"] if alignment else tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size (base): {base_size}")

    model, model_context_len = load_model(checkpoint_path, tokenizer.get_vocab_size(), device)
    if args.context_len == config.CONTEXT_LEN:
        args.context_len = model_context_len

    if args.context_len is None:
        args.context_len = model_context_len

    # Validate that the requested context length fits the trained model.
    if args.context_len > model_context_len:
        raise ValueError(
            f"--context-len ({args.context_len}) exceeds trained context length ({model_context_len})."
        )

    config.CONTEXT_LEN = args.context_len
    config.MAX_NEW_TOKENS = args.max_new_tokens
    config.TEMPERATURE = args.temperature
    config.TOP_K = args.top_k
    param_count, quantization = config.model_info(model)
    config.print_chat_hyperparams(param_count=param_count, quantization=quantization)

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
