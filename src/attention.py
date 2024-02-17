import torch
from torch import nn
from torch.nn import functional


class HeadAttention(nn.Module):
    def __init__(self, embedding_size, head_size, max_block_size, dropout):
        super().__init__()
        self.key_model = nn.Linear(embedding_size, head_size, bias=False)
        self.query_model = nn.Linear(embedding_size, head_size, bias=False)
        self.value_model = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(max_block_size, max_block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        _, block_size, _ = embeddings.shape  # B, T, C
        key = self.key_model(embeddings)  # B, T, 16
        query = self.query_model(embeddings)  # B, T, 16
        normalisation_factor = key.shape[-1] ** -0.5
        weights = query @ key.transpose(-2, -1) * normalisation_factor
        weights = weights.masked_fill(
            self.tril[:block_size, :block_size] == 0, value=float("-inf")
        )
        weights = functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        value = self.value_model(embeddings)  # B, T, 16

        return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embedding_size, head_size, max_block_size, dropout):
        super().__init__()
        self.multi_head = nn.ModuleList(
            [
                HeadAttention(embedding_size, head_size, max_block_size, dropout)
                for _ in range(num_heads)
            ]
        )
        self.projection = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.multi_head], dim=-1)
        x = self.projection(x)
        x = self.dropout(x)
        return x
