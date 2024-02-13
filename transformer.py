import torch
from torch import nn
from torch.nn import functional


class AttentionHead(nn.Module):
    def __init__(self, embedding_size, head_size, max_block_size):
        super().__init__()
        self.key_model = nn.Linear(embedding_size, head_size, bias=False)
        self.query_model = nn.Linear(embedding_size, head_size, bias=False)
        self.value_model = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(max_block_size, max_block_size))
        )

    def forward(self, embeddings):
        _, block_size, _ = embeddings.shape  # B, T, C
        key = self.key_model(embeddings)  # B, T, 16
        query = self.query_model(embeddings)  # B, T, 16
        normalisation_factor = key.shape[-1]**-0.5
        weights = query @ key.transpose(-2, -1) * normalisation_factor
        weights = weights.masked_fill(
            self.tril[:block_size, :block_size] == 0, value=float("-inf")
        )
        weights = functional.softmax(weights, dim=-1)
        value = self.value_model(embeddings)  # B, T, 16

        return weights @ value


class TransformerLanguageModel(nn.Module):
    def __init__(self, max_tokens, max_block_size, embedding_dimensions, head_size):
        super().__init__()
        self.max_block_size = max_block_size
        self.register_buffer("positions", torch.arange(max_block_size))
        self.token_embeddings = nn.Embedding(max_tokens, embedding_dimensions)
        self.position_embeddings = nn.Embedding(max_block_size, embedding_dimensions)
        self.attention_head = AttentionHead(
            embedding_size=embedding_dimensions,
            head_size=head_size,
            max_block_size=max_block_size,
        )
        self.language_model_head = nn.Linear(head_size, max_tokens)

    def forward(self, inputs, targets=None):
        inputs = inputs[:, -self.max_block_size :]
        batch_size, block_size = inputs.shape
        token_embeddings = self.token_embeddings(inputs)
        position_embeddings = self.position_embeddings(self.positions[:block_size])
        embeddings = token_embeddings + position_embeddings
        attention = self.attention_head(embeddings)
        logits = self.language_model_head(attention)
        if targets is None:
            loss = None
        else:
            batch_size, block_size, max_tokens = logits.shape
            logits_view = logits.view(batch_size * block_size, max_tokens)
            targets_view = targets.view(batch_size * block_size)
            loss = functional.cross_entropy(logits_view, targets_view)
        return logits, loss

    def generate(self, inputs, num_new_tokens):
        for _ in range(num_new_tokens):
            logits, _ = self(inputs)
            logits = logits[:, -1, :]
            prob_batch = functional.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(prob_batch, num_samples=1)
            inputs = torch.cat((inputs, next_tokens), dim=1)
        return inputs
