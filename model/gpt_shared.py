import torch


def rotate_half(x):
    # Swap and negate half dimensions for rotary mixing.
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def build_rope_cos_sin(inv_freq, positions, dtype):
    # Build cosine and sine tables for explicit absolute positions.
    freqs = torch.outer(positions.to(inv_freq.dtype), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def apply_rope_positions(rotary, q, k, positions):
    # Apply rotary embeddings using explicit positions for each query/key row.
    cos, sin = build_rope_cos_sin(rotary.inv_freq, positions, q.dtype)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def build_causal_mask(q_len, total_len, past_len, device):
    # Mark keys allowed for each query position in incremental decode.
    q_positions = past_len + torch.arange(q_len, device=device).unsqueeze(1)
    k_positions = torch.arange(total_len, device=device).unsqueeze(0)
    return k_positions <= q_positions


def project_qkv(attn, x):
    # Project to QKV and split the result into attention heads.
    batch_size, seq_len, _ = x.shape
    qkv = attn.qkv(x)
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
    return q, k, v


def merge_attention_heads(y, batch_size, seq_len, embed_size):
    # Merge attention heads back into the model dimension.
    return y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)
