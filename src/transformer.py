import torch
from torch import nn
from torch.nn import functional
from src.attention import MultiHeadAttention
from src.feedforward import FeedForward


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

    def get_next_token(self, inputs):
        logits, _ = self(inputs)
        logits = logits[:, -1, :]
        prob_batch = functional.softmax(logits, dim=-1)
        return torch.multinomial(prob_batch, num_samples=1)

    def generate(self, inputs, num_new_tokens, decode=None):
        for _ in range(num_new_tokens):
            next_token = self.get_next_token(inputs)
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs
