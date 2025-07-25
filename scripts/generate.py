import torch
from model.config import GPTConfig
# train.py
from tokenizer.tokenizer_utils import load_tokenizer
from training.logger_utils import create_log_dir

from tokenizer.tokenizer_utils import load_tokenizer, encode_dataset
from training.logger_utils import create_log_dir, ...

from model.gpt import GPT
from tokenizer.tokenizer_utils import load_tokenizer

# ----------- CONFIG + TOKENIZER -----------
cfg = GPTConfig(
    block_size=256,
    vocab_size=50304,
    n_layer=8,
    n_head=8,
    n_embd=512,
    rotary_embeddings=True,
    activation_function='swiglu',
    norm_type='rmsnorm',
    verbose=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPT(cfg).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()
print("\n✅ Model loaded.")

tokenizer = load_tokenizer("tokenizer/tokenizer_sentence.model")

# ----------- GENERATION FUNCTION -----------
def generate_text(prompt: str, max_tokens: int = 100, temperature: float = 1.0, top_k: int = 40):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)[0].tolist()
    decoded = tokenizer.decode(output_ids)
    return decoded

# ----------- SAMPLE RUN -----------
if __name__ == '__main__':
    prompt = input("\nEnter prompt: ")
    print("\nGenerating...\n")
    output = generate_text(prompt)
    print("---\n" + output + "\n---")