# Token-Based Learning Rate Schedule

## Summary

This design switches the learning-rate schedule from optimizer-step based
progress to token-based progress. The goal is to keep LR continuous when
training restarts with a different batch size, and to make LR depend on actual
training signal rather than batch shape.

## Goals

- LR does not jump when `BATCH_SIZE` changes between restarts.
- LR progress tracks actual tokens processed (including padding/packing effects).
- Reuse existing checkpoint counters (`total_tokens`).
- Keep logging and resume semantics clear and consistent.

## Non-Goals

- Redefine epochs as full dataset passes.
- Change dataset packing or sampling behavior.
- Add new user-facing configuration flags.

## Current Behavior

- Scheduler advances once per optimizer step.
- LR progress depends on `global_step`, which changes meaning with batch size.
- Progress `Total` is step-based (after recent changes).

## Proposed Behavior

- Scheduler uses cumulative token count as its progress variable.
- `total_tokens` from checkpoints becomes the LR source of truth.
- Progress `Total` reports token-based progress to match LR.
- `global_step` remains for visibility only.

## Computed Values

- `tokens_per_sample = CONTEXT_LEN - 1`
- `tokens_per_step = BATCH_SIZE * tokens_per_sample` (ideal, used for estimates)
- `estimated_total_tokens` from the token estimator
- `dataset_steps = ceil(estimated_total_tokens / tokens_per_step)` (informational)
- `target_tokens` = `MAX_TRAINING_FACTOR * param_count` (or full dataset if factor is 0)
- `target_steps = ceil(target_tokens / tokens_per_step)` (informational)

Token-based LR uses:

- `total_tokens_seen` for scheduler progress
- `target_tokens` as the schedule horizon

## Resume Behavior

- On resume, load `total_tokens` from the checkpoint.
- Align the scheduler to `total_tokens` before training continues.
- LR resumes at the exact same value regardless of batch size changes.

## Logging

Startup:

```
Dataset estimate: steps=... tokens=... tokens_per_step=...
Target:          epochs=... target_steps=... target_tokens=... (factor ...)
```

Progress:

```
Tokens 1.2m, Total 48.3%, Samples 38.4k, Epoch 1, Step 420, Global 420, Loss 2.3456, LR 6.00e-04, ...
```

`Total` is token-based to match the LR schedule.

## Implementation Plan

- Add a token-driven warmup+cosine scheduler variant.
- Replace per-step `scheduler.step()` calls with `scheduler.step(total_tokens_seen)`.
- Initialize `total_tokens_seen` from the checkpoint value.
- Update progress `Total` to be token-based and compute ETA from tokens.
- Keep `global_step` for display only.

## Compatibility

- No changes to checkpoint schema are required; `total_tokens` already exists.
- Existing checkpoints resume with correct LR alignment.
