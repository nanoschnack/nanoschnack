# Post-Training Chat Format

## Goal

Introduce a POST_TRAINING mode that expects chat-formatted input and masks loss
to assistant spans only. EOS resets the conversation. Training stops when the
loader ends.

## Input Format

Each record is a transcript using these tokens:

```
<|SYSTEM|> ... <|END|>
<|USER|> ... <|END|>
<|ASSISTANT|> ... <|END|>
```

The conversation reset marker is `<|EOS|>` (multiple conversations can appear
in one stream, split by EOS).

## Loss Masking Rules

When POST_TRAINING=1:

- Only tokens from `<|ASSISTANT|>` through `<|END|>` (inclusive) contribute to
  loss, and only when the preceding `<|USER|>` span is complete inside the same
  block.
- All other tokens are masked.
- If an `<|ASSISTANT|>` appears without a following `<|END|>`, everything after
  the unmatched `<|ASSISTANT|>` remains masked (conservative).
- If no assistant span is present, the entire sample is masked.

## Tokenization Flow

1) Encode each text record into `input_ids`.
2) Append `<|EOS|>` to each record.
3) Pack `input_ids` into fixed-length blocks.
4) Build a `loss_mask` for each packed block with the rules below.

## Block Boundary Injection Rules

When a packed block begins in the middle of a `<|USER|> ... <|END|>` span, the
block must be rewritten so it starts with the appropriate markers:

- If a complete `<|SYSTEM|> ... <|END|>` span has appeared in the current
  conversation, prepend that span at the start of the new block.
- Always prepend `<|USER|>` to avoid starting mid-span.
- The user span remains incomplete until its `<|END|>` is encountered inside
  the block, so any following `<|ASSISTANT|> ... <|END|>` in this block is
  masked unless a complete user span is present.
- If no system span has appeared in the conversation, do not inject one.

Conversations reset at `<|EOS|>`, so system/user history is cleared after EOS.

### Example

Given a conversation stream (S=system, U=user, A=assistant):

```
<|SYSTEM|> s0 <|END|>
<|USER|> u0 <|END|>
<|ASSISTANT|> a0 <|END|>
<|USER|> u1 <|END|>
<|ASSISTANT|> a1 <|END|>
<|USER|> u2 <|END|>
<|ASSISTANT|> a2 <|END|>
```

If a block boundary cuts inside the `u2` span, the next block is rewritten to
start with:

```
<|SYSTEM|> s0 <|END|>
<|USER|> u2 ...
```

The `<|ASSISTANT|> a2 <|END|>` span is masked unless the full `<|USER|> u2 <|END|>`
appears inside this same block.

If a block starts mid-user and still ends before `<|END|>`, stop packing the rest
of that document to avoid dangling assistant spans.

## Training Flow

- Shift targets for next-token prediction as usual.
- Shift the `loss_mask` to match target positions and apply it by setting
  masked targets to `-100`.
- Training ends when the loader ends; no extra epoch cycling.

## Configuration

- Add `POST_TRAINING=1` to enable assistant-only loss.
- Reuse `WARMUP_PCT` for the spec-change warmup reset.

## Dataset Generation Length Filter

When producing post-training chat text files, skip any tail that would exceed
the model context length:

- The first user/assistant pair must fit with the system span (if any).
- Subsequent pairs only need the user+assistant span to fit.
- If a pair does not fit, drop the rest of the conversation; keep the prefix.
- If the first pair does not fit, skip the conversation entirely.

## Tests

- Loader unit test: verify `loss_mask` is present and marks only assistant
  spans, including markers.
- Packing unit test: ensure `loss_mask` is packed and aligned with `input_ids`.
- Training unit test: masked tokens produce `-100` targets.
