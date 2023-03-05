import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets):
        # idx and targets are both (B, T) tensors of integers

        # arranges into a (B, T, C) tensor
        logits = self.token_embedding_table(idx)
        # where B is batch size, T is time, and C is channel

        # compute the loss
        # a god way to measure loss (quality of prediction) is the negative log likelyhood loss
        # this is the same as cross entropy loss

        # pytorch wants the ordering of logits to be (B, C, T)
        # so we need to transpose the logits tensor
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss
