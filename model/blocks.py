import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.config import GPTConfig

# -------------------- RMSNorm --------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True)
        return x / (norm + self.eps) * self.weight

# -------------------- SwiGLU MLP --------------------
class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden_dim = 4 * cfg.n_embd
        self.w1 = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.bias)
        self.proj = nn.Linear(hidden_dim, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.silu(self.w1(x)) * self.w2(x)))

# -------------------- Positional Encoding (RoPE) --------------------
def apply_rope(q, k):
    # Assumes q/k shape: [B, n_head, T, head_dim]
    B, H, T, D = q.size()
    half_dim = D // 2
    freq = torch.exp(
        -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=q.device) / half_dim
    )
    pos = torch.arange(0, T, dtype=torch.float32, device=q.device).unsqueeze(1)
    angles = pos * freq.unsqueeze(0)  # [T, half_dim]
    cos, sin = angles.cos(), angles.sin()

    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    q = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    return q, k

# -------------------- Causal Self-Attention --------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

        self.cfg = cfg
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head

        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

        if not cfg.use_flash:
            with torch.no_grad():
                mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
            self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).chunk(3, dim=-1)

        def reshape(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q, k, v = map(reshape, qkv)

        if self.cfg.rotary_embeddings:
            q, k = apply_rope(q, k)

        if self.cfg.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.cfg.dropout if self.training else 0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out_proj(y))

# -------------------- Transformer Block --------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        norm_cls = RMSNorm if cfg.norm_type == 'rmsnorm' else nn.LayerNorm
        self.norm1 = norm_cls(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = norm_cls(cfg.n_embd)
        self.mlp = SwiGLU(cfg) if cfg.activation_function == 'swiglu' else nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
