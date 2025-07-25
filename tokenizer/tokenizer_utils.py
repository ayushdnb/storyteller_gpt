import os
import sentencepiece as spm
import torch

def load_tokenizer(model_path: str) -> spm.SentencePieceProcessor:
    """
    Loads a SentencePiece tokenizer from a .model file.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Tokenizer model not found at: {model_path}")
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    print(f"[Tokenizer] Loaded from {model_path} | Vocab size: {sp.vocab_size()}")
    return sp

def encode_dataset(file_path: str, tokenizer: spm.SentencePieceProcessor) -> torch.Tensor:
    """
    Loads and encodes a text corpus using the tokenizer.
    Returns: Tensor of token ids.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    ids = tokenizer.encode(text, out_type=int)
    print(f"[Tokenizer] Encoded dataset | Tokens: {len(ids)}")
    return torch.tensor(ids, dtype=torch.long)

def decode_tokens(tokenizer: spm.SentencePieceProcessor, ids: list[int]) -> str:
    """
    Decodes a list of token ids back to text.
    """
    return tokenizer.decode(ids)
