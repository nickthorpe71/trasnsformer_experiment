import torch.nn as nn
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_embedding_dimensions: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        head_size = num_embedding_dimensions // num_heads
        self.self_attention = MultiHeadAttention(
            num_heads, num_embedding_dimensions, head_size, block_size, dropout)
        self.feed_forward = FeedForward(num_embedding_dimensions, dropout)
        self.layer_norm1 = nn.LayerNorm(num_embedding_dimensions)
        self.layer_norm2 = nn.LayerNorm(num_embedding_dimensions)

    def forward(self, x):
        x = x + self.self_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x
