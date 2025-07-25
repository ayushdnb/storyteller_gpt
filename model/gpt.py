import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.config import GPTConfig
from model.blocks import Block

class GPT(nn.Module):
    """
    Core GPT Language Model
    """
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if not cfg.rotary_embeddings:
            self.pos_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])

        norm_cls = nn.LayerNorm if cfg.norm_type == 'layernorm' else nn.Identity
        self.final_norm = norm_cls(cfg.n_embd)

        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight  # Weight tying

        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

        if cfg.verbose:
            print(f"Initialized GPT with {self.get_num_params():,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def get_num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.token_embedding.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.cfg.block_size, "Input sequence too long"

        token_embeddings = self.token_embedding(idx)
        pos_embeddings = 0
        if not self.cfg.rotary_embeddings:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_embeddings = self.pos_embedding(pos)[None, :, :]

        x = self.drop(token_embeddings + pos_embeddings)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        logits = self.lm_head(x if targets is not None else x[:, [-1]])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def configure_optimizers(self, device_type: str):
        decay, no_decay = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                (decay if p.dim() >= 2 else no_decay).append(p)

        optim_groups = [
            {"params": decay, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        fused_ok = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.learning_rate, betas=self.cfg.betas, fused=fused_ok)

        if self.cfg.verbose:
            print(f"[Optimizer] Using fused AdamW: {fused_ok}")

        return optimizer