try:
    from gpt_base import GPTBase
except ModuleNotFoundError:
    from .gpt_base import GPTBase


class GPT(GPTBase):
    """GPT-style decoder-only transformer for next-token prediction.

    Uses causal self-attention with pre-norm residual blocks.
    Returns logits for each token position in the sequence.
    Weight ties token embeddings and output projection.
    """

    def forward_blocks(self, x, attention_mask=None):
        # Run the residual transformer stack and final normalization.
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.ln(x)

    def forward(self, x, attention_mask=None):
        x = self.embed_inputs(x, start_pos=0)
        x = self.forward_blocks(x, attention_mask=attention_mask)
        return self.lm(x)
