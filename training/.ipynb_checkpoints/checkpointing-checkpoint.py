import os
import torch

def save_checkpoint(state: dict, path: str):
    """
    Saves model, optimizer, scaler, and scheduler state dicts.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"[Checkpoint] Saved to {path}")

def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scaler=None, scheduler=None):
    """
    Loads checkpoint and restores model, optimizer, scaler, and scheduler if provided.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scaler and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    if scheduler and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    print(f"[Checkpoint] Loaded from {path}")
    return checkpoint.get('step', 0)  # Default to 0 if step not saved
