# vLLM Transformers Wrapper (trust-remote-code)

This repo includes a minimal Transformers wrapper in `vllm_model/`.
It loads NanoSchnack checkpoints directly without converting to HF weights.

## Steps

1) Ensure a checkpoint exists (default: `checkpoints/latest.pt`).
2) Install the Transformers dependency:

```sh
uv pip install transformers
```

3) Launch vLLM with the local wrapper and GPT-2 tokenizer:

```sh
python -m vllm.entrypoints.openai.api_server \
  --model vllm_model \
  --trust-remote-code \
  --tokenizer openai-community/gpt2
```

## Notes

- `vllm_model/config.json` provides `auto_map` entries for `AutoModel` and
  `AutoModelForCausalLM`, which vLLM expects for custom models.
- `vllm_model/config.json` sets `checkpoint_path` to `../checkpoints/latest.pt`.
- Override the checkpoint path by editing `vllm_model/config.json` or passing a
  config with `checkpoint_path` to `from_pretrained`.
