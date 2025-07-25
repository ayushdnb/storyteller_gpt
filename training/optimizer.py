import torch
from torch.optim import AdamW
from model.config import GPTConfig

def configure_optimizer(model: torch.nn.Module, cfg: GPTConfig, device_type: str) -> torch.optim.Optimizer:
    """
    Sets up fused AdamW optimizer with correct parameter grouping.
    Separates decay and no_decay parameters (biases, norms).
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name or "Norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optim_groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    fused_ok = (device_type == "cuda") and ("fused" in AdamW.__init__.__code__.co_varnames)
    optimizer = AdamW(optim_groups, lr=cfg.learning_rate, betas=cfg.betas, fused=fused_ok)

    if cfg.verbose:
        print(f"[Optimizer] AdamW initialized | Fused: {fused_ok} | Decay params: {len(decay)} | No decay: {len(no_decay)}")

    return optimizer
