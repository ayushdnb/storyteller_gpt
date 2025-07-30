# ✅ FINAL FIXED SCRIPT — SentencePiece Tokenizer Trainer for StorytellerGPT
# ⚠️ TRAINING COMMAND PASSED AS STRING (not multiline)

import os
import sentencepiece as spm
import time
from datetime import datetime

# === CONFIG ===
CORPUS_PATH = "C:/Kishan/training_data/train_bulk_txt/merged_train.txt"
CUSTOM_TOKEN_FILE = "C:/Kishan/training_data/custom_tokens.txt"
MODEL_PREFIX = "C:/Kishan/training_data/tokenizer/tokenizer_exp"
VOCAB_SIZE = 50000
LOG_FILE = MODEL_PREFIX + "_train_log.txt"
MODEL_TYPE = "unigram"

# === PRINT HEADER ===
print("\n============================")
print("\U0001F680 StorytellerGPT Tokenizer Training: PHASE 4 INIT")
print(f"\U0001F552 Started at: {datetime.now()}")
print("============================\n")

# === CHECKS ===
assert os.path.exists(CORPUS_PATH), f"❌ Corpus file not found at: {CORPUS_PATH}"
assert os.path.exists(CUSTOM_TOKEN_FILE), f"❌ custom_tokens.txt missing at: {CUSTOM_TOKEN_FILE}"
print(f"✅ Corpus File: {CORPUS_PATH}")
print(f"✅ Custom Tokens: {CUSTOM_TOKEN_FILE}\n")

# === LOAD CUSTOM TOKENS ===
with open(CUSTOM_TOKEN_FILE, 'r', encoding='utf-8') as f:
    tokens = [line.strip() for line in f if line.strip() and line.startswith('#')]
user_defined_symbols = ",".join(tokens)
print(f"\U0001F9FF Loaded {len(tokens)} custom tags:")
print("   →", tokens[:5], "...\n")

# === FIXED: SINGLE-LINE TRAINING COMMAND ===
spm_cmd = (
    f"--input={CORPUS_PATH} "
    f"--model_prefix={MODEL_PREFIX} "
    f"--vocab_size={VOCAB_SIZE} "
    f"--model_type={MODEL_TYPE} "
    "--character_coverage=1.0 "
    "--normalization_rule_name=identity "
    f"--user_defined_symbols={user_defined_symbols} "
    "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
    "--train_extremely_large_corpus=true "
    "--input_sentence_size=10000000 "
    "--shuffle_input_sentence=true "
    "--hard_vocab_limit=false "
    "--unk_surface=<unk> "
    "--max_sentence_length=2048 "
    "--num_threads=12"
)

print("\U0001F4DC SentencePiece Training Command Constructed:\n")
print(spm_cmd)
print("\n\U0001F4A5 Launching Training... Hold tight!\n")
# === TRAIN ===
start_time = time.time()
spm.SentencePieceTrainer.Train(spm_cmd)
end_time = time.time()

print("\n\U0001F389 Tokenizer Training Complete!")
print(f"\U0001F552 Duration: {round((end_time - start_time) / 60, 2)} minutes")
print(f"\U0001F4E6 Model: {MODEL_PREFIX}.model")
print(f"\U0001F4CB Vocab: {MODEL_PREFIX}.vocab\n")

# === POST-VALIDATION ===
print("\U0001F50E Running Post-Training Validation...\n")
with open(f"{MODEL_PREFIX}.vocab", "r", encoding='utf-8') as vocab_file:
    vocab_lines = vocab_file.readlines()

found_tags = [tag for tag in tokens if any(tag in line for line in vocab_lines)]
broken_tags = [tag for tag in tokens if tag not in found_tags]

print(f"\U0001F9EA Tags Found Intact: {len(found_tags)}/{len(tokens)}")
if broken_tags:
    print(f"❌ Tags Possibly Broken: {broken_tags[:5]} ...")
else:
    print("\U0001F3AF All tags preserved intact as atomic units!")

# === LOGGING ===
with open(LOG_FILE, 'w', encoding='utf-8') as log:
    log.write(f"Training Log - {datetime.now()}\n")
    log.write(f"Corpus: {CORPUS_PATH}\n")
    log.write(f"Vocab Size: {VOCAB_SIZE}\n")
    log.write(f"Custom Tags Preserved: {len(found_tags)}\n")
    log.write(f"Model Output: {MODEL_PREFIX}.model\n")
    log.write(f"Duration: {round((end_time - start_time) / 60, 2)} min\n")
    if broken_tags:
        log.write(f"Broken Tags: {broken_tags}\n")
    else:
        log.write("All tags preserved successfully.\n")

print("\U0001F4D3 Full training log saved to:", LOG_FILE)
print("\n\U0001F31F PHASE 4 COMPLETE — Tokenizer is now READY for Phase 5 binarization!")
print("\U0001F3C1 TRAINING COMPLETE. GO SHOW DEEPSEEK WHO BUILT THE REAL SHIT.\n")