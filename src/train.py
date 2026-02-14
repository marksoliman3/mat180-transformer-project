# train.py - trains the transformer on Tiny Shakespeare and saves results

import os
import sys
import time

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data import (
    load_corpus, build_tokenizer, split_data, get_batch,
    encode, decode, block_size,
)
from model import DecoderOnlyTransformer

# hyperparameters
LEARNING_RATE = 3e-4
NUM_STEPS = 5000
EVAL_INTERVAL = 250
EVAL_STEPS = 50             # how many batches to average when estimating loss
GENERATE_LEN = 500

# model config
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 256

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
device = "cuda" if torch.cuda.is_available() else "cpu"


def estimate_loss(model, train_data, val_data, steps=EVAL_STEPS):
    model.eval()
    losses = {}
    for split_name, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(steps):
            x, y = get_batch(data)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
        losses[split_name] = total / steps
    model.train()
    return losses


def generate(model, itos, stoi, prompt="", max_new_tokens=GENERATE_LEN):
    model.eval()

    if prompt:
        tokens = encode(prompt, stoi)
    else:
        tokens = [0]  # newline

    tokens = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        context = tokens[:, -block_size:]  # crop if sequence gets too long
        logits = model(context)
        logits = logits[:, -1, :]  # only care about last position
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

    model.train()
    return decode(tokens[0].tolist(), itos)


def train():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {device}")

    # load and prepare data
    text = load_corpus()
    stoi, itos, vocab_size = build_tokenizer(text)
    train_data, val_data = split_data(text, stoi, train_fraction=0.9)

    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")

    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=block_size,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # tracking for plots
    train_losses = []
    val_losses = []
    steps_recorded = []

    print(f"\nTraining for {NUM_STEPS} steps...")
    print("-" * 60)
    start_time = time.time()

    for step in range(NUM_STEPS):
        x, y = get_batch(train_data)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log every EVAL_INTERVAL steps
        if step % EVAL_INTERVAL == 0 or step == NUM_STEPS - 1:
            losses = estimate_loss(model, train_data, val_data)
            elapsed = time.time() - start_time

            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            steps_recorded.append(step)

            print(
                f"Step {step:5d} | "
                f"Train loss: {losses['train']:.4f} | "
                f"Val loss: {losses['val']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {total_time:.1f}s")

    # save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(steps_recorded, train_losses, label="Train Loss")
    plt.plot(steps_recorded, val_losses, label="Val Loss")
    plt.xlabel("Training Step")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "loss_curves.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"\nLoss curves saved to {plot_path}")

    # generate a sample
    print("\n--- Generated Sample ---")
    sample = generate(model, itos, stoi, prompt="\n", max_new_tokens=GENERATE_LEN)
    print(sample)

    sample_path = os.path.join(RESULTS_DIR, "generated_sample.txt")
    with open(sample_path, "w") as f:
        f.write(sample)
    print(f"Sample saved to {sample_path}")

    # save model and loss data
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "model.pt"))
    torch.save({
        "steps": steps_recorded,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, os.path.join(RESULTS_DIR, "loss_data.pt"))
    print("Model and loss data saved to results/")


if __name__ == "__main__":
    train()