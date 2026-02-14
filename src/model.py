import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # each head works on a slice of the embedding
        self.scale = math.sqrt(self.head_dim)  # scaling factor for dot-product attention

        # Q, K, V projections — one big linear layer, then we split into heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # Output projection to combine heads back together
        self.out_proj = nn.Linear(d_model, d_model)

        # Causal mask: lower-triangular matrix that prevents attending to future tokens
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Project input into Q, K, V and split into heads
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # each is (B, T, C)

        # Reshape to (B, num_heads, T, head_dim) for parallel head computation
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V
        attn_scores = (q @ k.transpose(-2, -1)) / self.scale

        # Apply causal mask — fill future positions with -inf so softmax gives 0
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted sum of values
        out = attn_weights @ v  # (B, H, T, head_dim)

        # Concatenate heads and project back to d_model
        out = out.transpose(1, 2).contiguous().view(B, T, C) 
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Expand to higher dimension, apply nonlinearity, then project back
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        # Pre-norm: normalize before each sublayer, then add residual
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model = 64,
        num_heads = 4,
        num_layers = 2,
        d_ff = 256,
        max_seq_len = 128,
    ):
        super().__init__()
        self.d_model = d_model

        # Token embedding: maps each integer token to a d_model-dimensional vector
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Positional encoding: learned embedding for each position in the sequence
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Stack of decoder blocks
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, max_seq_len)
            for _ in range(num_layers)
        ])

        # Final layer norm and projection to vocabulary logits
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        # Initialize weights (helps with training stability)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape

        # Look up token embeddings and add positional embeddings
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        x = tok + pos

        # Pass through each decoder block
        for block in self.blocks:
            x = block(x)

        # Final norm + project to vocab size
        x = self.ln_final(x)
        logits = self.head(x)
        return logits