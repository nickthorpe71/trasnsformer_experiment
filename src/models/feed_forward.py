import torch.nn as nn


class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """

    def __init__(self, num_embedding_dimensions: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_embedding_dimensions, 4 * num_embedding_dimensions),
            nn.ReLU(),
            nn.Linear(4 * num_embedding_dimensions, num_embedding_dimensions),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
