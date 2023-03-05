import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ One head of self attention """

    def __init__(self, num_embedding_dimensions: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(num_embedding_dimensions, head_size, bias=False)
        self.query = nn.Linear(num_embedding_dimensions, head_size, bias=False)
        self.value = nn.Linear(num_embedding_dimensions, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # complete attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """ Multi-head attention """

    def __init__(self, num_heads: int, num_embedding_dimensions: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(num_embedding_dimensions, head_size, block_size, dropout) for _ in range(num_heads)])
        self.projection = nn.Linear(
            num_embedding_dimensions, num_embedding_dimensions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # dim-1 means we are concatonating over the channel dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.projection(out))
        return out
