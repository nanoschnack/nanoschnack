from transformers import PretrainedConfig


class NanoSchnackConfig(PretrainedConfig):
    """Configuration for the NanoSchnack GPT-style model.

    Mirrors model/config.py hyperparameters with minimal metadata.
    Stores defaults needed for local, trust-remote-code loading.
    """
    model_type = "nanoschnack"

    def __init__(
        self,
        vocab_size=0,
        embed_size=768,
        num_layers=12,
        num_heads=8,
        hidden_size=3072,
        context_len=1024,
        checkpoint_path=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.context_len = context_len
        self.checkpoint_path = checkpoint_path
