import os
import csv
from datetime import datetime


def create_log_dir(base_dir="logs") -> str:
    """
    Creates a timestamped log directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[Logger] Log dir created: {log_dir}")
    return log_dir

def create_csv_logger(log_dir: str, filename="losses.csv") -> csv.writer:
    """
    Initializes a CSV writer for logging training metrics.
    """
    path = os.path.join(log_dir, filename)
    f = open(path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "train_loss", "val_loss", "lr"])
    return writer, f

def log_sample(output_text: str, log_dir: str, step: int):
    """
    Appends a generated sample to a text file in outputs/.
    """
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", "samples.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n=== STEP {step} ===\n{output_text}\n")
    print(f"[Logger] Sample logged at step {step}")
