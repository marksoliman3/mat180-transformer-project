# generate.py - loads the trained transformer and generates text samples at different temperatures

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from data import load_corpus, build_tokenizer, encode, decode, block_size
from model import DecoderOnlyTransformer

D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 256

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
device = "cuda" if torch.cuda.is_available() else "cpu"


def generate(model, stoi, itos, prompt="\n", max_new_tokens=500, temperature=1.0):
    model.eval()

    tokens = encode(prompt, stoi)
    tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = tokens[:, -block_size:]
            logits = model(context)
            logits = logits[:, -1, :] / temperature  # scale logits before softmax
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

    return decode(tokens[0].tolist(), itos)


def main():
    print(f"Device: {device}\n")

    # rebuild tokenizer from the corpus
    text = load_corpus()
    stoi, itos, vocab_size = build_tokenizer(text)

    # load saved model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=block_size,
    ).to(device)

    model_path = os.path.join(RESULTS_DIR, "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))

    prompts = ["\n", "KING", "The "]
    temperatures = [0.5, 0.8, 1.0, 1.2]

    for prompt in prompts:
        for temp in temperatures:
            label = repr(prompt)
            print(f"Prompt: {label}  |  Temperature: {temp}")
            sample = generate(model, stoi, itos, prompt=prompt, temperature=temp)
            print(sample)
            print()


if __name__ == "__main__":
    main()