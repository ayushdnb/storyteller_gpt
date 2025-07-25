import os
import time
import torch
import random
import numpy as np
from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.tokenizer_utils import load_tokenizer, encode_dataset
from training.train_loop import train_loop
from training.optimizer import configure_optimizer
from training.scheduler import get_scheduler
from training.amp_utils import AMPScaler
from training.evaluation import evaluate, compute_perplexity
from training.checkpointing import save_checkpoint, load_checkpoint

# ---------------- ENV SETUP ----------------
SEED = 1337
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ---------------- CONFIG ----------------
cfg = GPTConfig(
    block_size=256,
    vocab_size=50304,
    n_layer=8,
    n_head=8,
    n_embd=512,
    dropout=0.15,
    rotary_embeddings=True,
    activation_function='swiglu',
    norm_type='rmsnorm',
    verbose=True
)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using {device}")

# ---------------- TOKENIZER + DATA ----------------
tokenizer = load_tokenizer("tokenizer/tokenizer_sentence.model")
data = encode_dataset("data/india_dataset.txt", tokenizer)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

# ---------------- BATCH LOADER ----------------
def get_batch(split: str, batch_size: int = 48, block_size: int = 256):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_) - block_size, (batch_size,))
    x = torch.stack([data_[i:i + block_size] for i in ix])
    y = torch.stack([data_[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

# ---------------- MODEL ----------------
model = GPT(cfg).to(device)
optimizer = configure_optimizer(model, cfg, device_type=device.type)
scheduler = get_scheduler(optimizer, num_steps=100_000, warmup_steps=1000)
scaler = AMPScaler()

# ---------------- TRAINING ----------------
run_dir = train_loop(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    cfg=cfg,
    get_batch=get_batch,
    num_steps=500_000,
    save_every=10_000,
    print_every=100,
    eval_batches=50,
    out_dir="checkpoints/"
)

print(f"\n✅ Training complete. Artifacts saved at: {run_dir}")
