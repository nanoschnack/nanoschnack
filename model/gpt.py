import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_layers=3, num_heads=8, hidden_size=2048, context_len=1024):
        super(GPT, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(context_len, embed_size)
        self.layers = nn.ModuleList([
            # Tranformer from "Attention is all you need" paper.
            # TODO(sttts): this is not what nanochat uses.
            nn.TransformerEncoderLayer(
                d_model=embed_size,
                nhead=num_heads,
                dim_feedforward=hidden_size,
                batch_first=True,  # Expect (B, T, E) from embeddings.
            )

            for _ in range(num_layers)
        ])
        self.output = nn.Linear(embed_size, vocab_size)

    def forward(self, x, attention_mask=None):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length).unsqueeze(0).to(x.device)
        x = self.token_embedding(x) + self.position_embedding(positions)

        src_key_padding_mask = None
        if attention_mask is not None:
            # True marks padded positions for TransformerEncoderLayer.
            src_key_padding_mask = attention_mask == 0

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)

        return self.output(x)
