from dataclasses import dataclass

@dataclass
class GPTConfig:
    """
    Master configuration for GPT model architecture and training hyperparameters.
    This config should be used consistently across the project.
    """

    # Architecture
    block_size: int = 256                 # Max sequence length (context window)
    vocab_size: int = 50304               # Tokenizer vocab size (SentencePiece or BPE)
    n_layer: int = 8                      # Number of transformer blocks
    n_head: int = 8                       # Number of self-attention heads
    n_embd: int = 512                     # Embedding dimension

    # Dropout rates
    dropout: float = 0.15                 # Dropout applied throughout model

    # Positional encoding
    rotary_embeddings: bool = True        # Use RoPE instead of learned embeddings

    # Attention
    bias: bool = True                     # Use bias in linear layers
    use_flash: bool = True                # Use FlashAttention if available

    # Normalization
    norm_type: str = 'rmsnorm'            # Options: 'layernorm', 'rmsnorm'

    # MLP
    activation_function: str = 'swiglu'   # Options: 'gelu', 'relu', 'swiglu'

    # Optimizer (only used in training)
    weight_decay: float = 0.1
    learning_rate: float = 1e-3
    betas: tuple = (0.9, 0.95)

    # Logging
    verbose: bool = True

    def print(self):
        print("\n====== GPT CONFIGURATION ======")
        for k, v in self.__dict__.items():
            print(f"{k:<20}: {v}")
        print("================================\n")

# Optional default config
if __name__ == '__main__':
    cfg = GPTConfig()
    cfg.print()
