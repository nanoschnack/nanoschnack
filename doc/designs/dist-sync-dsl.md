# Dist Sync DSL

## Goals
- Provide a declarative way to describe distributed synchronization needs.
- Reduce boilerplate and limit sequential collectives in training loops.
- Allow mixing collectives (all-reduce, all-gather, broadcast) and dtypes.

## Approach
- Use field metadata on dataclasses to declare sync semantics.
- Centralize packing and collective behavior in `model/sync.py`.
- Group fields by op, reduce type, and dtype to pack into fewer tensors.

## Field Semantics
- `all_reduce(reduce=...)` for scalars or small vectors.
- `all_gather(...)` for per-rank debug payloads.
- `broadcast(src=...)` for master-driven flags.
- Shapes can be fixed or callable for dynamic lengths.

## Call-Site Sketch
```python
@dataclass
class Synced:
    loss_sum: float = all_reduce("sum")
    token_count: float = all_reduce("sum")
    stop_flag: float = all_reduce("max")
    io_wait: float = all_reduce("max")
    counts: list[int] = all_reduce("sum", dtype="i64", shape=lambda: len(spec_keys))
    input_flag: int = broadcast(src=0, dtype="u8")

synced = Synced(
    loss_sum=macro_step.micro_loss_total,
    token_count=macro_step.micro_token_total,
    stop_flag=float(stop_requested),
    io_wait=macro_step.io_wait,
    counts=[source_row_counts.get(k, 0) for k in spec_keys],
    input_flag=int(is_master and input_request),
)

synced = sync(synced, device)
```

## Notes
- Core sync fields should stay in a single payload to reduce roundtrips.
- Debug-only gathers can be gated to avoid overhead on most steps.
- The DSL favors readability over micro-optimizing the pack/unpack logic.
