import torch
from torch.nn import functional as F
from typing import Callable

@torch.no_grad()
def evaluate(model, get_batch: Callable, eval_batches: int = 50, device: str = 'cuda') -> float:
    """
    Runs evaluation loop over validation data.
    Returns average loss per token.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for _ in range(eval_batches):
        xb, yb = get_batch('val')
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        total_loss += loss.item() * xb.size(0)
        total_tokens += xb.size(0)

    model.train()
    return total_loss / total_tokens

def compute_perplexity(loss: float) -> float:
    """
    Converts loss to perplexity.
    """
    return torch.exp(torch.tensor(loss)).item()
