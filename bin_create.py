# ==============================================
# ⚡ GODSPEED BIN+META CREATOR (v3.2) - DEEPSEEK EDITION
# Author: Cypher | Challenger to the Throne
# ==============================================

from __future__ import annotations
import os, sys, pickle, zlib, gc
from pathlib import Path
from datetime import datetime
from typing import Iterable, Tuple, List
import concurrent.futures
import threading

import sentencepiece as spm
import numpy as np
import pandas as pd

# ========== CONFIG ==========
BASE_DIR = Path("C:/Kishan/training_data")
TRAIN_PATH = BASE_DIR / "train_bulk_txt/merged_train.txt"
VAL_PATH = BASE_DIR / "val_bulk_txt/merged_val.txt"
TOKENIZER_MODEL = BASE_DIR / "tokenizer/tokenizer_exp.model"
CUSTOM_TOKEN_FILE = BASE_DIR / "custom_tokens.txt"
OUTPUT_DIR = BASE_DIR / "final_bin"

VOCAB_SIZE = 50_000
BLOCK_SIZE = 512
ENCODING = "utf-8"
CHUNK_SIZE = 32 * 1024 * 1024  # Reduced for better memory control
LOG_EVERY = 10_000_000
MAX_THREADS = min(4, os.cpu_count() or 4)  # Dynamic thread count
PRINT_EVERY_FILES = 100
MEMORY_BUFFER = 1024**3 * 2  # 2GB memory buffer

DTYPE = np.uint16 if VOCAB_SIZE < 65_536 else np.uint32
TOKEN_SIZE = DTYPE(0).itemsize

def validate_paths():
    required = [
        (TRAIN_PATH, "Train file missing"),
        (VAL_PATH, "Val file missing"),
        (TOKENIZER_MODEL, "Tokenizer model missing"),
        (CUSTOM_TOKEN_FILE, "custom_tokens.txt missing"),
    ]
    for p, msg in required:
        if not p.exists():
            sys.exit(f"❌ CRITICAL: {msg} at {p}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ All resources validated")

def load_tokenizer() -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    if not sp.Load(str(TOKENIZER_MODEL)):
        sys.exit(f"❌ Tokenizer failed to load: {TOKENIZER_MODEL}")
    print(f"✅ Tokenizer loaded | Vocab: {sp.vocab_size()} | Dtype: {DTYPE.__name__}")
    return sp

class QuantumTurboEncoder:
    def __init__(self, sp_model: spm.SentencePieceProcessor):
        self.sp = sp_model
        self.token_writer_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
    def _encode_chunk(self, chunk: bytes) -> Tuple[np.ndarray, int, int]:
        try:
            text = chunk.decode(ENCODING, errors="ignore")
            tokens = self.sp.encode(text, out_type=int)
            return np.array(tokens, dtype=DTYPE), len(text), zlib.crc32(chunk)
        except Exception as e:
            print(f"🚨 Encoding failed: {str(e)}")
            return np.array([], dtype=DTYPE), 0, 0

    def process_file(self, path: Path, label: str) -> Tuple[int, int, int]:
        total_bytes = path.stat().st_size
        out_path = OUTPUT_DIR / f"{label.lower()}.bin"
        crc_accumulator = 0
        token_count = 0
        char_count = 0
        bytes_processed = 0
        file_counter = 0
        
        print(f"🚀 Processing {label} | {total_bytes/1e9:.2f} GB")

        with (open(path, "rb") as f_in,
              open(out_path, "wb") as f_out,
              concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor):
            
            futures = []
            while True:
                # Memory control check
                if sys.getsizeof(gc.get_objects()) > (16 * 1024**3 - MEMORY_BUFFER):
                    gc.collect()
                
                chunk = f_in.read(CHUNK_SIZE)
                if not chunk:
                    break
                
                bytes_processed += len(chunk)
                crc_accumulator = zlib.crc32(chunk, crc_accumulator)
                futures.append(executor.submit(self._encode_chunk, chunk))
                file_counter += 1
                
                # Process completed futures
                while futures and (futures[0].done() or len(futures) >= MAX_THREADS):
                    future = futures.pop(0)
                    tokens, chars, chunk_crc = future.result()
                    
                    with self.token_writer_lock:
                        f_out.write(tokens.tobytes())
                    
                    with self.stats_lock:
                        token_count += tokens.size
                        char_count += chars
                    
                    # Logging
                    if token_count // LOG_EVERY != (token_count - tokens.size) // LOG_EVERY:
                        pct = 100 * bytes_processed / total_bytes
                        print(f"⚡ {token_count/1e6:.1f}M tokens | {pct:.1f}% | {bytes_processed//1e6:.0f} MB")
                
                if file_counter % PRINT_EVERY_FILES == 0:
                    print(f"📂 Processed {file_counter} chunks | {token_count/1e6:.2f}M tokens")
            
            # Process remaining futures
            for future in concurrent.futures.as_completed(futures):
                tokens, chars, chunk_crc = future.result()
                with self.token_writer_lock:
                    f_out.write(tokens.tobytes())
                with self.stats_lock:
                    token_count += tokens.size
                    char_count += chars

        # Final stats
        compression_ratio = bytes_processed / (token_count * TOKEN_SIZE) if token_count else 0
        print(f"✅ {label} complete → {token_count/1e6:.2f}M tokens | "
              f"Compression: {compression_ratio:.2f}x | CRC: {crc_accumulator:08X}")
        return bytes_processed, char_count, token_count

def validate_custom_tokens(sp: spm.SentencePieceProcessor) -> dict:
    custom_tags = []
    with open(CUSTOM_TOKEN_FILE, encoding=ENCODING) as f:
        for line in f:
            if token := line.strip():
                custom_tags.append(token)
    
    missing = [t for t in custom_tags if sp.piece_to_id(t) == -1]
    print(f"🎯 Custom tokens: {len(custom_tags)} | Missing: {len(missing)}")
    if missing:
        print("⚠️  Missing tokens:", ", ".join(missing[:5]))
        if len(missing) > 5:
            print(f"   ... and {len(missing)-5} more")
    return {t: sp.piece_to_id(t) for t in custom_tags if t not in missing}

def create_meta(sp, tokens_map, stats):
    meta = {
        "vocab_size": sp.vocab_size(),
        "block_size": BLOCK_SIZE,
        "tokenizer_path": str(TOKENIZER_MODEL),
        "special_token_ids": tokens_map,
        "dtype": DTYPE.__name__,
        "created_utc": datetime.utcnow().isoformat(),
        **stats
    }
    with open(OUTPUT_DIR / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)
    with open(OUTPUT_DIR / "meta.txt", "w", encoding=ENCODING) as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
    return meta

if __name__ == "__main__":
    validate_paths()
    sp = load_tokenizer()
    enc = QuantumTurboEncoder(sp)

    print("\n" + "="*50)
    tstat = enc.process_file(TRAIN_PATH, "train")
    print("\n" + "="*50)
    vstat = enc.process_file(VAL_PATH, "val")
    print("\n" + "="*50)

    tags = validate_custom_tokens(sp)
    stats = {
        "train_bytes": tstat[0],
        "train_chars": tstat[1],
        "train_tokens": tstat[2],
        "val_bytes": vstat[0],
        "val_chars": vstat[1],
        "val_tokens": vstat[2],
        "compression_ratio": round(tstat[0] / (tstat[2] * TOKEN_SIZE), 2)
    }
    meta = create_meta(sp, tags, stats)
    pd.DataFrame([meta]).to_csv(OUTPUT_DIR / "creation_log.csv", index=False)

    print("\n" + "="*50)
    print(f"🚀 Mission Accomplished | {stats['compression_ratio']:.2f}x compression")
    print(f"📆 Output: {OUTPUT_DIR}")
    print(f"🔥 Total Tokens: {(stats['train_tokens'] + stats['val_tokens'])/1e6:.1f}M")
    print("="*50)

