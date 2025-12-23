import torch
import torch.nn as nn


def causal_mask(seq_len, device):
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
        diagonal=1,
    )


class GPT(nn.Module):
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
        layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,  # Expect (B, T, E) from embeddings.
            norm_first=True, # Do what GPT-2 does: pre-norm instead of post-norm improving stability.
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers)
        self.ln = nn.LayerNorm(embed_size)
        self.lm = nn.Linear(embed_size, vocab_size, bias=False)

        # share weights between token embedding and output projection. Original
        # transformer paper and GPT-2 do this, and it improves performance.
        self.tok.weight = self.lm.weight # (B, T) parameters saved: 768*50k = 38M!

    def forward(self, x, attention_mask=None):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)
        x = self.tok(x) + self.pos(positions)

        padding_mask = None
        if attention_mask is not None:
            # True marks padded positions for TransformerEncoderLayer.
            padding_mask = attention_mask == 0
        causal = causal_mask(seq_length, x.device)

        x = self.blocks(x, mask=causal, src_key_padding_mask=padding_mask)
        x = self.ln(x)
        return self.lm(x)
