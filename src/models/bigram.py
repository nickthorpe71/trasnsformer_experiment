import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):

        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)

        return logits
