import torch
import torch.nn as nn
from torch.nn import functional as F
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size: int, num_embedding_dimensions: int, block_size: int, num_heads: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, num_embedding_dimensions)
        self.position_embedding_table = nn.Embedding(
            block_size, num_embedding_dimensions)
        self.self_attention_heads = MultiHeadAttention(
            num_heads, num_embedding_dimensions, num_embedding_dimensions // num_heads, block_size)  # ie. 4 heads of 8-dimensional self attention
        self.feed_forward = FeedForward(num_embedding_dimensions)
        self.language_modeling_head = nn.Linear(
            num_embedding_dimensions, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        B, T = idx.shape

        # arranges into a (B, T, C) tensor
        token_embeddings = self.token_embedding_table(idx)
        # where B is batch size, T is time, and C is channel
        positional_embeddings = self.position_embedding_table(
            torch.arange(T, device=idx.device))  # (T, C)
        x = token_embeddings + positional_embeddings  # (B, T, C)
        # apply one head of self attention (B, T, C)
        x = self.self_attention_heads(x)
        x = self.feed_forward(x)  # (B, T, C)
        logits = self.language_modeling_head(x)  # (B, T, vocab_size)

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

    def generate(self, idx, max_new_tokens: int, block_size: int):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last blck_size tokens
            idx_cont = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cont)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get the probabilities
            probs = F.softmax(logits, dim=-1)  # becomes (B, C)
            idx_next = torch.multinomial(
                probs, num_samples=1)  # becomes (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # becomes (B, T+1)
        return idx
