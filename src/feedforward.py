from torch import nn


class FeedForward(nn.Module):
    def __init__(self, dim, dropout, dim_multiplier=4):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * dim_multiplier),
            nn.ReLU(),
            nn.Linear(dim * dim_multiplier, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs):
        return self.feed_forward(inputs)
