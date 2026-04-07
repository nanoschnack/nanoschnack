# Inference

## Scope

This document defines the inference contract for serving NanoSchnack through vLLM first and later through NVIDIA Dynamo / Triton.

The first milestone is narrow on purpose:

- Autoregressive generation only.
- No retraining.
- No architecture change in the mathematical sense.
- `.pt` checkpoints in `models/` stay the source artifact.
- A custom vLLM model is the expected serving path.

## Supported Model Family

The currently supported checkpoint family is the RoPE-based NanoSchnack line in `models/`.

Supported checkpoint metadata today:

- `POS_EMBED_TYPE=rope`
- `ROPE_BASE=10000.0`
- `CONTEXT_LEN=1024`
- `EMBED_SIZE=768`
- `NUM_LAYERS=12`
- `NUM_HEADS=8`
- `HIDDEN_SIZE=3072`
- `VOCAB_SIZE=32256`

We intentionally do not commit to learned positional inference support just because the training code can instantiate it. The real pretrained snapshots in `models/` use RoPE.

## Supported Inputs

The model instance is defined by:

- Checkpoint filename in `models/`
- Tokenizer filename

Expected tokenizer filenames:

- `tokenizer.json`
- `tokenizer-v2.json`

Tokenizer selection is part of the model identity and must be explicit in inference.

## Checkpoint Rules

The serving loader must:

- Load `.pt` checkpoints directly.
- Read model metadata from checkpoint config when present.
- Support NanoSchnack checkpoints stored under `model`.
- Support `_orig_mod.`-prefixed parameter names.
- Reject unsupported positional modes early.

We keep `_orig_mod.` compatibility because several existing snapshots still use it.

## Tokenizer Rules

The inference tokenizer path must preserve the current runtime semantics:

- Load the selected base tokenizer JSON.
- Add the NanoSchnack special tokens when missing.
- Expand the vocabulary to the configured aligned size.
- Preserve token IDs exactly for a given tokenizer family.

Current checkpoint line:

- Existing `models/` snapshots use `tokenizer.json` by default.

Future model lines may use:

- `tokenizer-v2.json`

## Validation

The first validation target is greedy autoregressive parity against a PyTorch KV-cached reference path in `model/gpt_kv_cached.py`.

Minimum parity checks:

- Checkpoint loading succeeds for all supported snapshots.
- Tokenizer resolution produces the expected aligned vocab size.
- Greedy token-by-token generation matches the KV-cached reference path on fixed prompts.

## Non-Goals

- Learned positional inference support for the current pretrained line.
- General padded batched logits API.
- New export format replacing `.pt`.
- Quantization in the first milestone.
- Triton / Dynamo integration before vLLM parity is established.

## Tasks

- [x] Create `inference/` package for serving-specific code.
- [ ] Add inference spec loader for checkpoint config, tokenizer choice, and normalized state dict handling.
- [ ] Add inference tokenizer loader that reproduces NanoSchnack runtime vocab augmentation.
- [ ] Add an inference-only PyTorch KV-cached decode path in `model/gpt_kv_cached.py`.
- [ ] Add tests for `_orig_mod.` checkpoint normalization.
- [ ] Add tests for tokenizer family selection and aligned vocab size.
- [ ] Add greedy parity tests on fixed prompts.
- [ ] Prototype custom vLLM model loader for `.pt` checkpoints.
- [ ] Implement custom vLLM autoregressive generation path.
- [ ] Revisit Triton / Dynamo only after vLLM parity is stable.
