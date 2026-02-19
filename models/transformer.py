"""
Transformer text classifier from scratch for AG News.

No HuggingFace model imports. Built from Vaswani et al., "Attention Is All
You Need" (2017). Architecture: 6-layer encoder, 4 attention heads, 256-dim
hidden, learned positional encoding, [CLS] token pooling.

Target: >=91% clean accuracy on AG News test set.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.

    Splits input into num_heads parallel attention computations, each operating
    on d_model/num_heads dimensions. Scaled dot-product attention prevents
    gradient vanishing for large d_k.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(attn_output)


class TransformerBlock(nn.Module):
    """Single transformer encoder block: attention + FFN + LayerNorm + residual."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm architecture (more stable training than post-norm)
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        ffn_out = self.ffn(self.norm2(x))
        return x + ffn_out


class TextClassifier(nn.Module):
    """
    Transformer-based text classifier for AG News (4 classes).

    Architecture:
    - Learned token embeddings + positional embeddings
    - 6 TransformerBlocks (4 heads, 256 dim, 512 FFN)
    - [CLS] token pooling → linear classifier
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        d_ff: int = 512,
        max_seq_len: int = 256,
        num_classes: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # [CLS] token is at position 0
        cls_output = x[:, 0]
        return self.classifier(cls_output)

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract 256-d text representation. Used by CLIP-lite."""
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)[:, 0]
