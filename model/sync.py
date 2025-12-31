from dataclasses import field, fields

import torch
import torch.distributed as dist


_DTYPES = {
    "f16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "f64": torch.float64,
    "i32": torch.int32,
    "i64": torch.int64,
    "u8": torch.uint8,
    "u64": torch.uint64,
}

_REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}


def all_reduce(reduce, dtype="f32", shape=None):
    return field(metadata={"op": "all_reduce", "reduce": reduce, "dtype": dtype, "shape": shape})


def all_gather(dtype="f32", shape=None):
    return field(metadata={"op": "all_gather", "dtype": dtype, "shape": shape})


def broadcast(src=0, dtype="f32", shape=None):
    return field(metadata={"op": "broadcast", "src": src, "dtype": dtype, "shape": shape})


def _resolve_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    return _DTYPES[dtype]


def _resolve_shape(shape):
    if callable(shape):
        value = shape()
        if value is None:
            return None
        if isinstance(value, int):
            return (value,)
        return tuple(value)
    if shape is None:
        return None
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def _flatten_value(value, dtype, device):
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    return tensor.reshape(-1), tensor.shape


def _value_from_tensor(tensor, shape):
    if shape is None:
        shape = tensor.shape
    if tensor.numel() == 1:
        return tensor.item()
    return tensor.reshape(shape).tolist()


def sync(state, device):
    values = {sync_field.name: getattr(state, sync_field.name) for sync_field in fields(state)}
    groups = {}
    for sync_field in fields(state):
        meta = sync_field.metadata
        if not meta:
            continue
        op = meta.get("op")
        dtype = _resolve_dtype(meta.get("dtype", "f32"))
        shape = _resolve_shape(meta.get("shape"))
        if op == "all_reduce":
            key = (op, meta.get("reduce", "sum"), dtype)
        elif op == "all_gather":
            key = (op, dtype)
        elif op == "broadcast":
            key = (op, meta.get("src", 0), dtype)
        else:
            raise ValueError(f"Unsupported sync op: {op}")
        groups.setdefault(key, []).append((sync_field, shape, dtype))
        values[sync_field.name] = getattr(state, sync_field.name)

    for key, group in groups.items():
        op = key[0]
        flat_tensors = []
        field_meta = []
        offset = 0
        for sync_field, shape, dtype in group:
            flat, value_shape = _flatten_value(values[sync_field.name], dtype, device)
            size = flat.numel()
            flat_tensors.append(flat)
            field_meta.append((sync_field, shape or value_shape, size, offset))
            offset += size
        packed = torch.cat(flat_tensors) if len(flat_tensors) > 1 else flat_tensors[0]

        if op == "all_reduce":
            reduce = key[1]
            dist.all_reduce(packed, op=_REDUCE_OPS[reduce])
            for sync_field, shape, size, offset in field_meta:
                slice_tensor = packed[offset:offset + size]
                values[sync_field.name] = _value_from_tensor(slice_tensor, shape)

        elif op == "broadcast":
            src = key[1]
            dist.broadcast(packed, src=src)
            for sync_field, shape, size, offset in field_meta:
                slice_tensor = packed[offset:offset + size]
                values[sync_field.name] = _value_from_tensor(slice_tensor, shape)

        elif op == "all_gather":
            world_size = dist.get_world_size()
            gathered = [torch.zeros_like(packed) for _ in range(world_size)]
            dist.all_gather(gathered, packed)
            for sync_field, shape, size, offset in field_meta:
                field_values = [
                    gathered_tensor[offset:offset + size].reshape(shape)
                    for gathered_tensor in gathered
                ]
                values[sync_field.name] = field_values

    return type(state)(**values)
