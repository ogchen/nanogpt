import argparse
import torch

MAX_TOKENS = 255  # Allow usage of uint8
TRAINING_DATA_PERCENTAGE = 0.9
BLOCK_SIZE = 8
BATCH_SIZE = 4


def generate_batch(data):
    assert len(data) >= BLOCK_SIZE
    offsets = torch.randint(len(data) - BLOCK_SIZE + 1, (BATCH_SIZE,))
    input_batch = torch.stack([data[offset:offset + BLOCK_SIZE] for offset in offsets])
    target_batch = torch.stack([data[offset + 1: offset + 1 + BLOCK_SIZE] for offset in offsets])
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
    decode = lambda encodings: "".join((encoding_to_token.get(e, "") for e in encodings))

    return encode, decode


def extract_token_universe(text, max_tokens):
    return sorted(list(set(text)))[:max_tokens]


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
    token_universe = extract_token_universe(content, MAX_TOKENS)
    encode, decode = generate_encoder_decoder(token_universe)
    data = torch.tensor(encode(content), dtype=torch.uint8)
    train_data, val_data = split_data(data, TRAINING_DATA_PERCENTAGE)
    input, target = generate_batch(train_data)
    print("input", input)
    print("target", target)


if __name__ == "__main__":
    main()
