# MAT 180 — Linear Algebra Foundations of Decoder-Only Transformer Architecture

**Course:** MAT 180: Special Topics in Mathematics  
**Track:** Algorithm  

## Overview

This project analyzes the linear algebra foundations underlying decoder-only transformer architecture. We implement a minimal transformer from scratch in PyTorch, focusing on the mathematical mechanics of multi-head self-attention and feed-forward networks.

## Repository Structure

```
mat180-transformer-project/
├── src/
│   ├── data.py           # Data pipeline (tokenization, batching)
│   ├── model.py          # Decoder-only transformer implementation
│   └── train.py          # Training loop, evaluation, and text generation
├── data/                 # Tiny Shakespeare dataset (auto-downloaded)
├── results/              # Loss curves, generated samples, saved model
└── README.md
```

## Usage

### Training
```bash
python src/train.py
```

This will:
- Download the Tiny Shakespeare dataset (if not already present)
- Train the model for 5000 steps (~7 min on CPU)
- Save loss curves, a generated text sample, and model weights to `results/`

### Outputs
All outputs are saved to `results/`:
- `loss_curves.png` — training and validation loss plot
- `generated_sample.txt` — sample Shakespeare-like text from the trained model
- `model.pt` — saved model weights
- `loss_data.pt` — raw loss data for re-plotting

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
- Strang, G. (2019). *Linear Algebra and Learning from Data.* Wellesley-Cambridge Press.