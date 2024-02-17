class Tokenizer:
    def __init__(self, text):
        self.tokens = sorted(list(set(text)))
        self.token_to_encoding = {c: i for i, c in enumerate(self.tokens)}
        self.encoding_to_token = {i: c for i, c in enumerate(self.tokens)}

    def encode(self, text):
        return [self.token_to_encoding[t] for t in text if t in self.tokens]

    def decode(self, encodings):
        return "".join((self.encoding_to_token.get(e, "") for e in encodings))

    def count(self):
        return len(self.tokens)
