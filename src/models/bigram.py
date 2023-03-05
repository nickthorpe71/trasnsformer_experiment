import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers

        # arranges into a (B, T, C) tensor
        logits = self.token_embedding_table(idx)
        # where B is batch size, T is time, and C is channel

        if targets is None:
            return logits, None

        # compute the loss
        # a good way to measure loss (quality of prediction) is the negative log likelyhood loss
        # this is the same as cross entropy loss

        # pytorch wants the ordering of logits to be (B, C, T)
        # so we need to transpose the logits tensor
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)  # becomes (B, C)
            idx_next = torch.multinomial(
                probs, num_samples=1)  # becomes (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # becomes (B, T+1)
        return idx
