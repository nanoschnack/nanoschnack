# Token-Balanced Interleave (Least-Consumed)

## Summary

We interleave datasets by always sampling from the source with the fewest
observed tokens so far. Per-spec token counts are tracked during training
and persisted in checkpoints for accurate resumes. This removes estimators
and weight updates in favor of a simple least-consumed policy.

## Goals

- Keep dataset sampling balanced by real token consumption.
- Use actual token counts instead of estimator-based averages.
- Preserve resume behavior with backward-compatible checkpoints.
- Minimize notebook changes and keep logging accurate.

## Non-Goals

- Adapting a global training token budget mid-run.
- Reintroducing adaptive weight recomputation.
- Changing packing, shuffling, or tokenizer behavior.

## Current Behavior

- Packed datasets are interleaved by a least-consumed token rule.
- Per-spec row counts and token counts are tracked during training.
- Checkpoints store row and token offsets per spec, including old specs.
- Dataset position logs use real row/token counts (no token estimates).

## Interleave Policy

Maintain a token counter for each active dataset spec. On each sample:

- Select the dataset with the smallest token count.
- Break ties randomly with a fixed seed.
- Increment the selected dataset count by the sample token length.

This keeps token consumption balanced across sources without weights.

## Checkpointing

`resume_state` stores both row and token offsets per spec:

```
{"datasets": [{"spec": "...", "row_offset": 123, "token_offset": 456}, ...]}
```

Resume behavior:

- Row and token offsets are loaded for all specs.
- Unknown specs in a checkpoint are preserved.
- Missing token offsets default to 0 for current specs.

## Implementation Notes

- `build_interleaved_dataset` accepts `token_counts` and uses a
  least-consumed generator.
- Packed datasets no longer force `with_format("torch")` to avoid
  redundant tensor conversion warnings.
- Dataset position logs include only real row/token counts.

## Risks and Mitigations

- **New spec dominance on resume:** new specs start at token count 0 and
  can temporarily dominate. (Planned: seed new specs at a baseline.)
- **Token-count drift across ranks:** counts are synchronized via
  per-rank reductions at logging/checkpoint boundaries.

## Testing Plan

- Unit test for least-consumed interleave ordering.
- Resume test to ensure token offsets persist and load correctly.
- Smoke test to confirm dataset position logging and no tensor warnings.
