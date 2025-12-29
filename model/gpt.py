import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal multi-head attention using PyTorch SDPA.

    Uses a single QKV projection and scaled dot-product attention.
    Supports optional padding masks while enforcing causality.
    Targets Flash Attention when CUDA supports it.
    """
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        if embed_size % num_heads != 0:
            raise ValueError("embed_size must be divisible by num_heads.")
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.qkv = nn.Linear(embed_size, 3 * embed_size)
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        batch_size, seq_len, embed_size = x.shape

        # Project to QKV and split into attention heads.
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Expand padding mask to (B, 1, 1, T) for SDPA broadcasting.
        attn_mask = None
        if attention_mask is not None:
            # Expect integer/bool padding mask with 1 for tokens and 0 for padding.
            if torch.is_floating_point(attention_mask):
                raise TypeError("attention_mask must be integer/bool with 1 for tokens and 0 for padding.")
            min_val = int(attention_mask.min().item())
            max_val = int(attention_mask.max().item())
            if min_val < 0 or max_val > 1:
                raise ValueError("attention_mask values must be 0 or 1.")
            attn_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)

        # Run causal attention and project back to the model dimension.
        dropout_p = self.dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=True,
            dropout_p=dropout_p,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_size)
        y = self.proj(y)
        return self.dropout(y)


class FeedForward(nn.Module):
    """Position-wise feed-forward network for transformer blocks.

    Expands embeddings to a hidden size, applies GELU, and projects back.
    Includes dropout on both projections for regularization.
    """
    def __init__(self, embed_size, hidden_size, dropout):
        super().__init__()
        self.input = nn.Linear(embed_size, hidden_size)
        self.output = nn.Linear(hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.output(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with causal self-attention.

    Applies LayerNorm before attention and MLP blocks.
    Uses residual connections around both sublayers.
    Preserves GPT-style training dynamics.
    """
    def __init__(self, embed_size, num_heads, hidden_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.attn = CausalSelfAttention(embed_size, num_heads, dropout)
        self.mlp = FeedForward(embed_size, hidden_size, dropout)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT-style decoder-only transformer for next-token prediction.

    Uses causal self-attention with pre-norm residual blocks.
    Returns logits for each token position in the sequence.
    Weight ties token embeddings and output projection.
    """
    def __init__(
        self,
        vocab_size,
        embed_size=768,
        num_layers=12,
        num_heads=8,
        hidden_size=4*768,
        context_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, embed_size)
        self.pos = nn.Embedding(context_len, embed_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_size, num_heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embed_size)
        self.lm = nn.Linear(embed_size, vocab_size, bias=False)
        # Share weights between token embedding and output projection.
        self.tok.weight = self.lm.weight

        # Apply GPT-2 initialization after tying weights.
        self._init_weights(num_layers)

    def _init_weights(self, num_layers):
        # Apply GPT-2 initialization for all module types.
        self.apply(self._init_module_weights)

        # Scale GPT-2 residual projections below the base 0.02 std to control residual growth.
        # The attention/MLP outputs are added back into the residual stream, so large
        # projections compound over layers and can destabilize training without scaling.
        scaled_std = 0.02 / math.sqrt(2 * num_layers)
        for block in self.blocks:
            block.attn.proj.weight.data.normal_(mean=0.0, std=scaled_std)
            block.mlp.output.weight.data.normal_(mean=0.0, std=scaled_std)

    @staticmethod
    def _init_module_weights(module):
        # Use GPT-2 linear init: normal(0, 0.02) instead of PyTorch's Kaiming uniform.
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            # Use zero biases instead of PyTorch's uniform bias init.
            if module.bias is not None:
                module.bias.data.zero_()

        # Use GPT-2 embedding init: normal(0, 0.02) instead of PyTorch's normal(0, 1).
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

        # Keep LayerNorm weights at 1 and biases at 0, matching PyTorch defaults.
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, attention_mask=None):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.tok(x) + self.pos(positions)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln(x)
        return self.lm(x)
