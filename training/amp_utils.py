import torch
from torch.cuda.amp import autocast, GradScaler

class AMPContext:
    """
    Context manager for mixed precision autocasting.
    Automatically switches to CPU autocast if CUDA is unavailable.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"

    def __enter__(self):
        self.ctx = autocast(device_type=self.device_type, enabled=self.enabled)
        return self.ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.ctx.__exit__(exc_type, exc_val, exc_tb)


class AMPScaler:
    """
    Wraps GradScaler functionality with fallback for CPU.
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.enabled)

    def scale(self, loss):
        return self.scaler.scale(loss) if self.enabled else loss

    def step(self, optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        if self.enabled:
            self.scaler.update()

    def unscale_(self, optimizer):
        if self.enabled:
            self.scaler.unscale_(optimizer)

    def state_dict(self):
        return self.scaler.state_dict() if self.enabled else {}

    def load_state_dict(self, state):
        if self.enabled:
            self.scaler.load_state_dict(state)
