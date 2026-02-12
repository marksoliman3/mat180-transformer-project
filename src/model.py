"""
Decoder-Only Transformer — Minimal Implementation

A from-scratch PyTorch implementation emphasizing the linear algebra
operations at each stage: multi-head self-attention, feed-forward
networks, layer normalization, and residual connections.

References:
    Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
    Strang, G. (2019). Linear Algebra and Learning from Data.
"""

import torch
import torch.nn as nn
import math


# TODO: Implement each component with explicit linear algebra operations.
# The goal is clarity over efficiency — we want to expose the matrix
# multiplications, projections, and normalizations that make transformers work.


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Key linear algebra operations:
        - Linear projections: Q = XW_Q, K = XW_K, V = XW_V
        - Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V
        - Output projection: concatenated heads multiplied by W_O
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: Initialize projection matrices W_Q, W_K, W_V, W_O
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # TODO: Implement attention with explicit matrix operations
        pass


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network.

    Key linear algebra operations:
        - Two affine transformations: W_2 * ReLU(W_1 * x + b_1) + b_2
        - W_1 expands dimensionality, W_2 projects back down
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # TODO: Initialize weight matrices W_1, W_2 and biases
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement feed-forward with explicit matrix multiplications
        pass


class LayerNorm(nn.Module):
    """Layer normalization.

    Key linear algebra operation:
        - Normalizes across the feature dimension: (x - mean) / std
        - Followed by learned affine transform: gamma * x_norm + beta
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # TODO: Initialize gamma (scale) and beta (shift) parameters
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Implement normalization
        pass


class DecoderBlock(nn.Module):
    """Single decoder block: self-attention + FFN with residual connections.

    Residual connection: output = LayerNorm(x + Sublayer(x))
    Each residual connection preserves the original signal while adding
    the learned transformation — a key architectural choice.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        # TODO: Compose attention, FFN, and layer norms
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # TODO: Implement with residual connections
        pass


class DecoderOnlyTransformer(nn.Module):
    """Minimal decoder-only transformer model.

    Architecture:
        Token embedding + positional encoding
        → N x DecoderBlock
        → Final layer norm
        → Linear output projection
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 128,
    ):
        super().__init__()
        # TODO: Initialize embeddings, decoder blocks, and output projection
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Full forward pass
        pass
