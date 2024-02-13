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


class TransformerBlock(nn.Module):
    def __init__(self, max_block_size, embedding_dimensions, num_heads, dropout):
        super().__init__()
        self.multi_attention = MultiHeadAttention(
            num_heads=num_heads,
            embedding_size=embedding_dimensions,
            head_size=embedding_dimensions // num_heads,
            max_block_size=max_block_size,
            dropout=dropout,
        )
        self.feed_forward = FeedForward(embedding_dimensions, dropout)
        self.layer_norm1 = nn.LayerNorm(embedding_dimensions)
        self.layer_norm2 = nn.LayerNorm(embedding_dimensions)

    def forward(self, x):
        x = x + self.multi_attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        max_tokens,
        max_block_size,
        embedding_dimensions,
        num_heads,
        num_transformer_blocks,
        dropout,
    ):
        super().__init__()
        self.max_block_size = max_block_size
        self.register_buffer("positions", torch.arange(max_block_size))
        self.token_embeddings = nn.Embedding(max_tokens, embedding_dimensions)
        self.position_embeddings = nn.Embedding(max_block_size, embedding_dimensions)
        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    max_block_size=max_block_size,
                    embedding_dimensions=embedding_dimensions,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_transformer_blocks)
            ],
            nn.LayerNorm(embedding_dimensions)
        )
        self.language_model_head = nn.Linear(embedding_dimensions, max_tokens)

    def forward(self, inputs, targets=None):
        inputs = inputs[:, -self.max_block_size :]
        batch_size, block_size = inputs.shape
        token_embeddings = self.token_embeddings(inputs)
        position_embeddings = self.position_embeddings(self.positions[:block_size])
        x = token_embeddings + position_embeddings
        x = self.transformer_blocks(x)
        logits = self.language_model_head(x)
        if targets is None:
            loss = None
        else:
            batch_size, block_size, max_tokens = logits.shape
            logits_view = logits.view(batch_size * block_size, max_tokens)
            targets_view = targets.view(batch_size * block_size)
            loss = functional.cross_entropy(logits_view, targets_view)
        return logits, loss

    def generate(self, inputs, num_new_tokens, decode=None):
        for _ in range(num_new_tokens):
            logits, _ = self(inputs)
            logits = logits[:, -1, :]
            prob_batch = functional.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(prob_batch, num_samples=1)
            if decode:
                print(decode(next_tokens[0].tolist()), end="")
            inputs = torch.cat((inputs, next_tokens), dim=1)
        return inputs
