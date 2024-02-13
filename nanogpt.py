import argparse
import torch
from bigram import BigramLanguageModel

MAX_TOKENS = 255  # Allow usage of uint8
TRAINING_DATA_PERCENTAGE = 0.9
BLOCK_SIZE = 8
BATCH_SIZE = 4


def train_model(train_data, model, optimizer, decode, iterations):
    for i in range(iterations):
        inputs, targets = sample_batch(train_data)
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            generated = model.generate(
                torch.zeros(1, 1, dtype=torch.long), num_new_tokens=100
            )
            print(f"iteration {i}, loss {loss.item()}: {decode(generated[0].tolist())}")


def sample_batch(data):
    assert len(data) >= BLOCK_SIZE
    offsets = torch.randint(len(data) - BLOCK_SIZE + 1, (BATCH_SIZE,))
    input_batch = torch.stack(
        [data[offset : offset + BLOCK_SIZE] for offset in offsets]
    )
    target_batch = torch.stack(
        [data[offset + 1 : offset + 1 + BLOCK_SIZE] for offset in offsets]
    )
    return input_batch, target_batch


def split_data(data, training_data_percentage):
    train_size = int(training_data_percentage * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]
    return train_data, val_data


def generate_encoder_decoder(token_list):
    token_to_encoding = {c: i for i, c in enumerate(token_list)}
    encoding_to_token = {i: c for i, c in enumerate(token_list)}

    encode = lambda tokens: [token_to_encoding[t] for t in tokens if t in token_list]
    decode = lambda encodings: "".join(
        (encoding_to_token.get(e, "") for e in encodings)
    )

    return encode, decode


def extract_token_universe(text, max_tokens=None):
    token_universe = sorted(list(set(text)))
    return token_universe[:max_tokens] if max_tokens else token_universe


def read_file(filename):
    with open(filename, mode="r") as f:
        text = f.read()
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        prog="nanogpt", description="Executes a GPT language model"
    )
    parser.add_argument("filename")
    return parser.parse_args()


def main():
    args = parse_args()
    content = read_file(args.filename)
    token_universe = extract_token_universe(content)
    encode, decode = generate_encoder_decoder(token_universe)
    data = torch.tensor(encode(content), dtype=torch.long)
    train_data, val_data = split_data(data, TRAINING_DATA_PERCENTAGE)
    model = BigramLanguageModel(len(token_universe))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_model(train_data, model, optimizer, decode, iterations=100000)


if __name__ == "__main__":
    main()
