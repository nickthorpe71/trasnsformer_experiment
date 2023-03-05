import torch.nn as nn
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_embedding_dimensions: int, num_heads: int, block_size: int):
        super().__init__()
        head_size = num_embedding_dimensions // num_heads
        self.self_attention = MultiHeadAttention(
            num_heads, num_embedding_dimensions, head_size, block_size)
        self.feed_forward = FeedForward(num_embedding_dimensions)

    def forward(self, x):
        x = x + self.self_attention(x)
        x = x + self.feed_forward(x)
        return x
