import argparse
import os
import textwrap

import codecs
import torch

try:
    from model import setup_paths
except ModuleNotFoundError:
    from __init__ import setup_paths

import config
from device import device_info, pick_device, print_device_info
from gpt import GPT
from gpt_kv_cached import GPTKVCached
from checkpoint_metadata import load_checkpoint_metadata
from checkpointer import (
    load_checkpoint_config,
    load_model_state_dict,
    resize_vocab_state_dict,
)
from tokenizer import ASSISTANT_TOKEN, END_TOKEN, EOS_TOKEN, USER_TOKEN, load_tokenizer, print_vocab_alignment


def load_model(checkpoint_path, vocab_size, device):
    # Construct the model and optionally load weights from checkpoint.
    state_dict = None
    ckpt_vocab_size = None

    if checkpoint_path is not None:
        metadata = load_checkpoint_metadata(checkpoint_path, map_location=device)
        if metadata is not None:
            ckpt_vocab_size = metadata.vocab_size
            state_dict = metadata.state_dict

    model = GPTKVCached(
        vocab_size=vocab_size,
        embed_size=config.EMBED_SIZE,
        num_layers=config.NUM_LAYERS,
        num_heads=config.NUM_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        context_len=config.CONTEXT_LEN,
        dropout=0.0,
        pos_embed_type=config.POS_EMBED_TYPE,
        rope_base=config.ROPE_BASE,
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


def _build_id_to_bytes(tokenizer):
    vocab = tokenizer.get_vocab()
    max_id = max(vocab.values())
    id_to_token = [None] * (max_id + 1)
    for token, idx in vocab.items():
        id_to_token[idx] = token

    byte_decoder = _build_byte_decoder()
    id_to_bytes = [None] * (max_id + 1)
    for idx, token in enumerate(id_to_token):
        if token is None:
            continue
        id_to_bytes[idx] = bytes(byte_decoder[ch] for ch in token)
    return id_to_bytes


def _build_byte_decoder():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = {b: chr(c) for b, c in zip(bs, cs)}
    return {v: k for k, v in byte_encoder.items()}


def _cache_status_label(use_cache):
    # Keep cache mode wording consistent across startup and REPL commands.
    return "enabled" if use_cache else "disabled"


def _checkpoint_status_label(checkpoint_path):
    # Print the resolved checkpoint path or an explicit random-init marker.
    if checkpoint_path is None:
        return "none (random init)"
    return str(checkpoint_path)


DEFAULT_SAMPLE_PROMPT = "Der Sinn des Lebens ist was?"


def _build_prompt(user_text, history=""):
    # Build the current-mode prompt with optional chat history.
    if config.POST_TRAINING:
        if not history:
            history = EOS_TOKEN
        return f"{history}{USER_TOKEN}{user_text}{END_TOKEN}{ASSISTANT_TOKEN}"
    return f"{EOS_TOKEN}{user_text}"


def generate_reply_stream(
    model,
    tokenizer,
    prompt,
    context_len,
    max_new_tokens,
    temperature,
    top_k,
    device,
    stop_id=None,
    use_cache=False,
):
    # Stream tokens from autoregressive decoding for a single reply.
    prompt_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([prompt_ids[-context_len:]], device=device, dtype=torch.long)
    id_to_bytes = _build_id_to_bytes(tokenizer)
    decoder = codecs.getincrementaldecoder("utf-8")()
    if stop_id is None:
        stop_id = tokenizer.token_to_id(EOS_TOKEN)
    use_byte_decoder = True

    with torch.no_grad():
        cache = None
        if use_cache and hasattr(model, "forward_cached"):
            logits, cache = model.forward_cached(input_ids)
        else:
            logits = model(input_ids)
        for _ in range(max_new_tokens):
            next_id = sample_next_token(logits[:, -1, :].squeeze(0), temperature, top_k)
            if stop_id is not None and next_id == stop_id:
                break

            # Prefer byte-level streaming but fall back if bytes are invalid UTF-8.
            if use_byte_decoder and next_id < len(id_to_bytes) and id_to_bytes[next_id] is not None:
                try:
                    chunk = decoder.decode(id_to_bytes[next_id], final=False)
                except UnicodeDecodeError:
                    use_byte_decoder = False
                    chunk = tokenizer.decode([next_id])
            else:
                chunk = tokenizer.decode([next_id])

            if chunk:
                yield chunk
            next_input = torch.tensor([[next_id]], device=device)

            # Keep a sliding token window for both cached and uncached decode.
            input_ids = torch.cat([input_ids, next_input], dim=1)
            if input_ids.size(1) > context_len:
                input_ids = input_ids[:, -context_len:]

            if use_cache and hasattr(model, "forward_cached"):
                # Rebuild cache from the trimmed window once we slide past the
                # context limit so cache length and position semantics stay in
                # sync with the plain full-sequence path.
                if cache is not None and input_ids.size(1) == context_len:
                    if hasattr(model, "cache_seq_len"):
                        cache_len = model.cache_seq_len(cache[0]) if cache else 0
                    else:
                        cache_len = input_ids.size(1)
                    if cache_len >= context_len:
                        cache = None
                        logits, cache = model.forward_cached(input_ids)
                        continue
                logits, cache = model.forward_cached(next_input, cache)
            else:
                logits = model(input_ids)


def _sample_replies(
    model,
    tokenizer,
    prompt,
    context_len,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    device,
    stop_id,
    use_cache,
):
    # Generate multiple fresh-context replies for the same prompt.
    replies = []
    for _ in range(num_samples):
        replies.append(
            "".join(
                generate_reply_stream(
                    model,
                    tokenizer,
                    prompt,
                    context_len=context_len,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    device=device,
                    stop_id=stop_id,
                    use_cache=use_cache,
                )
            )
        )
    return replies


def _stream_text_chunks(chunks):
    # Stream reply chunks with the same wrapping and newline behavior as chat replies.
    pending_backslash = False
    current_line = ""
    first_line = True
    wrapper = textwrap.TextWrapper(
        width=80,
        break_long_words=False,
        break_on_hyphens=False,
    )

    def flush_buffer(final=False):
        nonlocal current_line, first_line
        if not current_line:
            return
        lines = wrapper.wrap(current_line) or [""]
        if final:
            to_print = lines
            current_line = ""
        else:
            if len(lines) <= 1:
                return
            to_print = lines[:-1]
            current_line = lines[-1]
        for line in to_print:
            if first_line:
                print(line, end="", flush=True)
                first_line = False
            else:
                print("\n  " + line, end="", flush=True)

    def start_new_line():
        nonlocal first_line
        flush_buffer(final=True)
        print("\n  ", end="", flush=True)
        first_line = False

    for token in chunks:
        for char in token:
            if pending_backslash:
                if char == "n":
                    start_new_line()
                    pending_backslash = False
                    continue
                print("\\", end="", flush=True)
                pending_backslash = False
            if char == "\\":
                pending_backslash = True
                continue
            if char == "\n":
                start_new_line()
                continue
            current_line += char
            flush_buffer()
    if pending_backslash:
        current_line += "\\"
    flush_buffer(final=True)


def _handle_repl_command(user_text, temperature, top_k, show_tokens, use_cache, model):
    # Handle interactive runtime commands and return updated settings.
    if user_text == "/help":
        return True, (
            "Commands: /help, /quit, /exit, /reset, /temp <value>, /topk <value>, "
            "/sample <prompt>, "
            "/debug on|off, /cache [on|off]"
        ), temperature, top_k, show_tokens, use_cache

    if user_text.startswith("/temp"):
        parts = user_text.split(maxsplit=1)
        if len(parts) != 2:
            return True, f"Temperature: {temperature}", temperature, top_k, show_tokens, use_cache
        try:
            temperature = float(parts[1])
        except ValueError:
            return True, "Temperature must be a number.", temperature, top_k, show_tokens, use_cache
        return True, f"Temperature set to {temperature}", temperature, top_k, show_tokens, use_cache

    if user_text.startswith("/topk"):
        parts = user_text.split(maxsplit=1)
        if len(parts) != 2:
            return True, f"Top-k: {top_k}", temperature, top_k, show_tokens, use_cache
        try:
            top_k = int(parts[1])
            if top_k < 0:
                raise ValueError
        except ValueError:
            return True, "Top-k must be a non-negative integer.", temperature, top_k, show_tokens, use_cache
        return True, f"Top-k set to {top_k}", temperature, top_k, show_tokens, use_cache

    if user_text.startswith("/debug"):
        parts = user_text.split(maxsplit=1)
        if len(parts) == 1:
            show_tokens = not show_tokens
        elif parts[1] in {"on", "off"}:
            show_tokens = parts[1] == "on"
        else:
            return True, "Usage: /debug [on|off]", temperature, top_k, show_tokens, use_cache
        return True, f"Debug tokens {'enabled' if show_tokens else 'disabled'}.", temperature, top_k, show_tokens, use_cache

    if user_text.startswith("/cache"):
        if not hasattr(model, "forward_cached"):
            return True, "KV cache is not supported by this model.", temperature, top_k, show_tokens, use_cache
        parts = user_text.split(maxsplit=1)
        if len(parts) == 1:
            use_cache = not use_cache
            return True, f"KV cache {_cache_status_label(use_cache)}.", temperature, top_k, show_tokens, use_cache
        if parts[1] not in {"on", "off"}:
            return True, "Usage: /cache [on|off]", temperature, top_k, show_tokens, use_cache
        use_cache = parts[1] == "on"
        return True, f"KV cache {_cache_status_label(use_cache)}.", temperature, top_k, show_tokens, use_cache

    if user_text.startswith("/"):
        return True, f"Unknown command: {user_text}", temperature, top_k, show_tokens, use_cache

    return False, None, temperature, top_k, show_tokens, use_cache


def run_repl(model, tokenizer, context_len, max_new_tokens, temperature, top_k, device):
    # Enable basic readline navigation if available.
    try:
        import readline  # noqa: F401
    except Exception:
        pass

    debug_level = int(os.getenv("DEBUG", "0"))
    show_tokens = False
    use_cache = hasattr(model, "forward_cached")
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
        if user_text in {"/quit", "/exit"}:
            break
        if user_text == "/reset":
            print("History cleared.")
            history = ""
            continue

        if user_text.startswith("/sample"):
            parts = user_text.split(maxsplit=1)
            sample_text = parts[1] if len(parts) == 2 else DEFAULT_SAMPLE_PROMPT
            sample_prompt = _build_prompt(sample_text, history="")
            sample_stop_id = tokenizer.token_to_id(END_TOKEN if config.POST_TRAINING else EOS_TOKEN)
            for index in range(1, 11):
                print(f"{index}. ", end="", flush=True)
                _stream_text_chunks(
                    generate_reply_stream(
                        model,
                        tokenizer,
                        sample_prompt,
                        context_len=context_len,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        device=device,
                        stop_id=sample_stop_id,
                        use_cache=use_cache,
                    )
                )
                print()
            continue

        handled, message, temperature, top_k, show_tokens, use_cache = _handle_repl_command(
            user_text,
            temperature,
            top_k,
            show_tokens,
            use_cache,
            model,
        )
        if handled:
            if message:
                print(message)
            continue

        prompt = _build_prompt(user_text, history=history)

        if show_tokens:
            print(f"tokens> {tokenizer.encode(user_text).ids}")
        print("bot> ", end="", flush=True)
        pending_backslash = False
        current_line = ""
        first_line = True
        wrapper = textwrap.TextWrapper(
            width=80,
            break_long_words=False,
            break_on_hyphens=False,
        )

        def flush_buffer(final=False):
            nonlocal current_line, first_line
            if not current_line:
                return
            lines = wrapper.wrap(current_line) or [""]
            if final:
                to_print = lines
                current_line = ""
            else:
                if len(lines) <= 1:
                    return
                to_print = lines[:-1]
                current_line = lines[-1]
            for line in to_print:
                if first_line:
                    print(line, end="", flush=True)
                    first_line = False
                else:
                    print("\n  " + line, end="", flush=True)

        def start_new_line():
            nonlocal first_line
            flush_buffer(final=True)
            print("\n  ", end="", flush=True)
            first_line = False
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
                stop_id=tokenizer.token_to_id(END_TOKEN if config.POST_TRAINING else EOS_TOKEN),
                use_cache=use_cache,
            ):
                for char in token:
                    if pending_backslash:
                        if char == "n":
                            start_new_line()
                            pending_backslash = False
                            continue
                        print("\\", end="", flush=True)
                        pending_backslash = False
                    if char == "\\":
                        pending_backslash = True
                        continue
                    if char == "\n":
                        start_new_line()
                        continue
                    current_line += char
                    flush_buffer()
                reply_parts.append(token)
        except KeyboardInterrupt:
            print()
        if pending_backslash:
            current_line += "\\"
        flush_buffer(final=True)
        if reply_parts:
            print()
        if debug_level >= 1 and reply_parts:
            raw_context = prompt + "".join(reply_parts)
            print(f"raw> {raw_context}")
        if config.POST_TRAINING and reply_parts:
            assistant_text = "".join(reply_parts)
            history += f"{USER_TOKEN}{user_text}{END_TOKEN}{ASSISTANT_TOKEN}{assistant_text}"
            if not history.endswith(END_TOKEN):
                history += END_TOKEN

            # Keep a rolling context window to avoid unbounded growth.
            history_ids = tokenizer.encode(history).ids
            if len(history_ids) > context_len:
                history = tokenizer.decode(history_ids[-context_len:], skip_special_tokens=False)


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

    # Apply checkpoint config before loading tokenizer so TOKENIZER_FILENAME is correct.
    load_checkpoint_config(checkpoint_path)
    tokenizer = load_tokenizer()
    print_vocab_alignment(tokenizer)
    model, model_context_len = load_model(checkpoint_path, tokenizer.get_vocab_size(), device)
    mode = "chat" if config.POST_TRAINING else "completion"
    print("Chat:")
    print(f"  mode={mode}")
    print(f"  checkpoint={_checkpoint_status_label(checkpoint_path)}")
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
    config.print_chat_hyperparams(
        param_count=param_count,
        quantization=quantization,
        kv_cache=_cache_status_label(hasattr(model, "forward_cached")),
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
