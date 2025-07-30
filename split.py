import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
import re
import shutil
import gc
import time
from contextlib import suppress
from concurrent.futures import ThreadPoolExecutor, as_completed

# ====== CONFIG ======
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(r"C:/Users/ayush/OneDrive/Desktop/Stories")
CLEANED_DIR = BASE_DIR / "cleaned_files_tagged"
SPLIT_DIR = BASE_DIR / "split"
TRAIN_DIR = SPLIT_DIR / "train"
VAL_DIR = SPLIT_DIR / "val"
REJECTED_DIR = SPLIT_DIR / "rejected"
LOG_DIR = SPLIT_DIR / "split_logs"
MERGE_ROOT = Path(r"C:/Kishan/training_data")
TRAIN_BULK = MERGE_ROOT / "train_bulk_txt/merged_train.txt"
VAL_BULK = MERGE_ROOT / "val_bulk_txt/merged_val.txt"

for d in [TRAIN_DIR, VAL_DIR, REJECTED_DIR, LOG_DIR, TRAIN_BULK.parent, VAL_BULK.parent]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "split_pipeline.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# ====== CONSTANTS & PATTERNS ======
DOC_PATTERN = re.compile(r'(.*)', re.DOTALL)  # Match whole content
TOKEN_PATTERN = re.compile(r'(#[a-z_]+|\b\w{2,}\b)')
MIN_TOKENS = 5

# ====== UTILS ======
def copy_task(src, dest):
    try:
        shutil.copy(src, dest)
        return True, src.name, None
    except Exception as e:
        return False, src.name, str(e)

def parallel_copy(file_paths, target_dir, max_workers=8):
    logger.info(f"🛠️ Copying {len(file_paths)} files to {target_dir.name}")
    start_time = time.time()
    tasks = [(src, target_dir / src.name) for src in file_paths]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(copy_task, src, dest) for src, dest in tasks]
        success_count = 0
        for i, future in enumerate(as_completed(futures)):
            try:
                success, filename, error = future.result()
                if success:
                    success_count += 1
                else:
                    logger.error(f"❌ Failed: {filename} - {error}")
                    with suppress(Exception):
                        shutil.copy(CLEANED_DIR / filename, REJECTED_DIR / filename)
            except Exception as e:
                logger.exception(f"🔥 Future crash: {str(e)}")

            if (i + 1) % 100 == 0:
                logger.info(f"⏱️ Copied {i+1}/{len(tasks)} files")

    logger.info(f"📦 Finished copying {success_count}/{len(file_paths)} files in {time.time()-start_time:.2f}s")
    return success_count

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
            match = DOC_PATTERN.search(file_content)
            if not match:
                return None, file_path, 'missing_DOC_TAG'
            content = match.group(1)
            tokens = TOKEN_PATTERN.findall(content)
            if len(tokens) < MIN_TOKENS:
                return None, file_path, 'low_token_count'
            return content, file_path, len(tokens)
    except Exception as e:
        return None, file_path, str(e)

def merge_bulk_files(file_paths, output_path, label):
    with open(output_path, 'w', encoding='utf-8') as bulk_file:
        for i, file in enumerate(file_paths):
            try:
                text = file.read_text(encoding='utf-8', errors='ignore')
                bulk_file.write(text.strip() + "\n\n")
                if (i + 1) % 100 == 0:
                    logger.info(f"🧩 [{label}] Merged {i+1}/{len(file_paths)} files")
            except Exception as e:
                logger.error(f"🔥 Merge failed for {file.name}: {e}")

# ====== MAIN ======
def main():
    global_start = time.time()
    logger.info("🚀 Starting stratified split + bulk merge pipeline")

    files = list(CLEANED_DIR.glob("*_clean.txt"))
    logger.info(f"📂 Total files found: {len(files)}")

    valid_texts = []
    valid_paths = []
    token_counts = []
    rejected_paths = []

    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = {executor.submit(process_file, f): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            text, file_path, result = future.result()

            if text is None:
                rejected_paths.append(file_path)
            else:
                valid_texts.append(text)
                valid_paths.append(file_path)
                token_counts.append(result)

            if (i + 1) % 100 == 0:
                logger.info(f"🔄 Processed {i+1}/{len(files)} files")

    logger.info(f"📊 Valid: {len(valid_paths)} | Rejected: {len(rejected_paths)}")
    if not valid_paths:
        logger.warning("🛑 No valid files. Exiting.")
        return

    logger.info("⚙️ Vectorizing + SVD")
    X = HashingVectorizer(n_features=2**16, ngram_range=(1, 2), alternate_sign=False).transform(valid_texts)
    X_reduced = TruncatedSVD(n_components=48, random_state=RANDOM_SEED).fit_transform(X)

    logger.info("🧠 Clustering")
    kmeans = MiniBatchKMeans(n_clusters=min(50, max(8, len(valid_paths)//100)), random_state=RANDOM_SEED)
    labels = kmeans.fit_predict(X_reduced)

    df = pd.DataFrame({ 'path': valid_paths, 'cluster': labels, 'tokens': token_counts })
    train_paths, val_paths = [], []
    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster].sample(frac=1, random_state=RANDOM_SEED)
        split_idx = int(len(cluster_df) * 0.9)
        train_paths.extend(cluster_df.iloc[:split_idx]['path'])
        val_paths.extend(cluster_df.iloc[split_idx:]['path'])

    logger.info(f"✅ Stratified split done: Train={len(train_paths)} | Val={len(val_paths)}")
    parallel_copy(train_paths, TRAIN_DIR)
    parallel_copy(val_paths, VAL_DIR)

    logger.info("📦 Merging into bulk .txt files")
    merge_bulk_files(train_paths, TRAIN_BULK, "TRAIN")
    merge_bulk_files(val_paths, VAL_BULK, "VAL")

    logger.info("🏁 Pipeline complete in %.2fs", time.time() - global_start)

if __name__ == "__main__":
    main()
