import argparse

import torch

try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths

from config import CONTEXT_LEN, MAX_NEW_TOKENS, TEMPERATURE, TOP_K
from device import pick_device
from gpt import GPT
from tokenizer import load_tokenizer


def load_model(checkpoint_path, vocab_size, context_len, device):
    # Construct the model and optionally load weights from checkpoint.
    model = GPT(vocab_size=vocab_size, context_len=context_len).to(device).eval()

    if checkpoint_path is None:
        return model

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    return model


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


def generate_reply(model, tokenizer, prompt, context_len, max_new_tokens, temperature, top_k, device):
    # Run autoregressive decoding for a single reply.
    prompt_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([prompt_ids[-context_len:]], device=device, dtype=torch.long)

    new_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)[:, -1, :].squeeze(0)
            next_id = sample_next_token(logits, temperature, top_k)
            new_ids.append(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
            if input_ids.size(1) > context_len:
                input_ids = input_ids[:, -context_len:]

    return tokenizer.decode(new_ids).strip()


def run_repl(model, tokenizer, context_len, max_new_tokens, temperature, top_k, device):
    # Enable basic readline navigation if available.
    try:
        import readline  # noqa: F401
    except Exception:
        pass

    history = ""
    print("Type '/quit' to exit.")

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in {"/quit", "/exit"}:
            break

        history += f"User: {user_text}\nAssistant:"
        reply = generate_reply(
            model,
            tokenizer,
            history,
            context_len=context_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        print(f"bot> {reply}")
        history += f" {reply}\n"


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

    # Validate that the requested context length fits the trained model.
    if args.context_len > CONTEXT_LEN:
        raise ValueError(
            f"--context-len ({args.context_len}) exceeds trained context length ({CONTEXT_LEN})."
        )

    device = pick_device()
    tokenizer = load_tokenizer()
    model = load_model(checkpoint_path, tokenizer.get_vocab_size(), CONTEXT_LEN, device)

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
