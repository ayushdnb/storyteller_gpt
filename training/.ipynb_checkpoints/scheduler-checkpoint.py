import math
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler(optimizer, num_steps: int, warmup_steps: int = 1000):
    """
    Returns a cosine decay scheduler with warmup.
    Warmup linearly increases LR from 0 to base LR.
    Cosine decay follows after warmup.
    """
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
