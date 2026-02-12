# MAT 180 — Linear Algebra Foundations of Decoder-Only Transformer Architecture

**Course:** MAT 180: Special Topics in Mathematics 
**Track:** Algorithm  
**Team:** Mark Soliman, Yahir De Leon, Benedict Lim

## Overview

This project analyzes the linear algebra foundations underlying decoder-only transformer architecture. We implement a minimal transformer from scratch in PyTorch, focusing on the mathematical mechanics of multi-head self-attention, feed-forward networks, residual connections, and layer normalization.

## Repository Structure

```
mat180-transformer-project/
├── src/                  # Source code
│   ├── model.py          # Decoder-only transformer implementation
│   ├── train.py          # Training loop
│   └── evaluate.py       # Evaluation and metrics
├── notebooks/            # Jupyter notebooks for exploration and visualization
├── docs/                 # Report drafts and presentation materials
├── results/              # Output plots, tables, and saved results
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/mat180-transformer-project.git
   cd mat180-transformer-project
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
```bash
python src/train.py
```

### Evaluation
```bash
python src/evaluate.py
```

### Outputs
- Trained model checkpoints are saved to `results/`
- Plots and tables are saved to `results/`

## References

- Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS.
- Strang, G. (2019). *Linear Algebra and Learning from Data.* Wellesley-Cambridge Press.
