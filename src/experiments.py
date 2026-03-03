# experiments.py - runs hyperparameter experiments and saves comparison plots

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
    load_corpus, build_tokenizer, split_data, get_batch, block_size,
)
from model import DecoderOnlyTransformer

NUM_STEPS = 3000
EVAL_INTERVAL = 150
EVAL_STEPS = 50
LEARNING_RATE = 3e-4

# baseline config
BASELINE = {
    "d_model": 64,
    "num_heads": 4,
    "num_layers": 2,
    "d_ff": 256,
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
device = "cuda" if torch.cuda.is_available() else "cpu"


def estimate_loss(model, train_data, val_data):
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        total = 0.0
        for _ in range(EVAL_STEPS):
            x, y = get_batch(data)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total += loss.item()
        losses[name] = total / EVAL_STEPS
    model.train()
    return losses


def train_one_config(vocab_size, config, train_data, val_data):
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_len=block_size,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    steps_recorded = []
    train_losses = []
    val_losses = []

    start = time.time()

    for step in range(NUM_STEPS):
        x, y = get_batch(train_data)
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == NUM_STEPS - 1:
            losses = estimate_loss(model, train_data, val_data)
            steps_recorded.append(step)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])

            elapsed = time.time() - start
            print(
                f"  Step {step:5d} | "
                f"Train: {losses['train']:.4f} | "
                f"Val: {losses['val']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

    total_time = time.time() - start
    return {
        "steps": steps_recorded,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "time": total_time,
        "params": param_count,
    }


def plot_experiment(title, param_name, results, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, res in results.items():
        ax1.plot(res["steps"], res["train_losses"], label=f"{param_name}={label}")
        ax2.plot(res["steps"], res["val_losses"], label=f"{param_name}={label}")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{title} - Train Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.set_title(f"{title} - Val Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(path, dpi=150)
    plt.close()


def run_experiment(name, param_key, values, vocab_size, train_data, val_data):

    results = {}
    for val in values:
        config = BASELINE.copy()
        config[param_key] = val
        label = str(val)

        print(f"\n{param_key}={val} | d_model={config['d_model']}, "
              f"heads={config['num_heads']}, layers={config['num_layers']}, "
              f"d_ff={config['d_ff']}")

        results[label] = train_one_config(vocab_size, config, train_data, val_data)

        r = results[label]
        print(f"  -> {r['params']:,} params, "
              f"val loss {r['final_val_loss']:.4f}, "
              f"{r['time']:.1f}s")

    return results


def print_summary(all_results):
    print(f"\n{'Experiment':<25} {'Setting':<10} {'Params':>10} "
          f"{'Train Loss':>12} {'Val Loss':>12} {'Time (s)':>10}")

    for exp_name, results in all_results.items():
        for label, res in results.items():
            print(f"{exp_name:<25} {label:<10} {res['params']:>10,} "
                  f"{res['final_train_loss']:>12.4f} {res['final_val_loss']:>12.4f} "
                  f"{res['time']:>10.1f}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {device}")
    print(f"Steps per run: {NUM_STEPS}")
    print(f"Eval every {EVAL_INTERVAL} steps")

    text = load_corpus()
    stoi, itos, vocab_size = build_tokenizer(text)
    train_data, val_data = split_data(text, stoi, train_fraction=0.9)

    print(f"Vocab size: {vocab_size}")
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens:   {len(val_data):,}")

    all_results = {}

    # experiment 1: vary attention heads
    exp1 = run_experiment(
        "Attention Heads", "num_heads", [1, 2, 4],
        vocab_size, train_data, val_data,
    )
    plot_experiment("Attention Heads", "heads", exp1, "exp_heads.png")
    all_results["Attention Heads"] = exp1

    # experiment 2: vary decoder layers
    exp2 = run_experiment(
        "Decoder Layers", "num_layers", [1, 2, 4],
        vocab_size, train_data, val_data,
    )
    plot_experiment("Decoder Layers", "layers", exp2, "exp_layers.png")
    all_results["Decoder Layers"] = exp2

    # experiment 3: vary embedding dimension
    exp3 = run_experiment(
        "Embedding Dim", "d_model", [32, 64, 128],
        vocab_size, train_data, val_data,
    )
    plot_experiment("Embedding Dim", "d_model", exp3, "exp_dmodel.png")
    all_results["Embedding Dim"] = exp3

    print_summary(all_results)

    torch.save(all_results, os.path.join(RESULTS_DIR, "experiment_results.pt"))
    print(f"\nResults saved to results/experiment_results.pt")


if __name__ == "__main__":
    main()