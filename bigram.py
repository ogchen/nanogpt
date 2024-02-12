import torch
from torch import nn
from torch.nn import functional


class BigramLanguageModel(nn.Module):
    def __init__(self, max_tokens):
        super().__init__()
        self.logits_lookup = nn.Embedding(max_tokens, max_tokens)

    def forward(self, inputs, targets=None):
        logits = self.logits_lookup(inputs)
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
