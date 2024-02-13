import torch
import time
import argparse
from functools import wraps
from bigram import BigramLanguageModel

MAX_TOKENS = 255  # Allow usage of uint8
TRAINING_DATA_PERCENTAGE = 0.9
BLOCK_SIZE = 8
BATCH_SIZE = 4
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
EVALUATION_ITERATIONS = 200


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution of {func.__name__} took {end_time - start_time} seconds")
        return result

    return wrapper


@torch.no_grad
def estimate_loss(data, model):
    model.eval()
    losses = torch.zeros(EVALUATION_ITERATIONS)
    for k in range(EVALUATION_ITERATIONS):
        inputs, targets = sample_batch(data)
        _, loss = model(inputs, targets)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def print_training_status(iteration, train_data, val_data, model, decode):
    train_loss = estimate_loss(train_data, model)
    val_loss = estimate_loss(val_data, model)
    generated = model.generate(
        torch.zeros(1, 1, dtype=torch.long, device=DEVICE), num_new_tokens=100
    )
    print(f"iteration: {iteration}, training: {train_loss}, val loss: {val_loss}")
    print("sample output:")
    print(decode(generated[0].tolist()))
    print()


@timing_decorator
def train_model(train_data, val_data, model, optimizer, decode, iterations):
    for i in range(iterations):
        inputs, targets = sample_batch(train_data)
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 5000 == 0:
            print_training_status(i, train_data, val_data, model, decode)


def sample_batch(data):
    assert len(data) >= BLOCK_SIZE + 1
    offsets = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    input_batch = torch.stack(
        [data[offset : offset + BLOCK_SIZE] for offset in offsets]
    )
    target_batch = torch.stack(
        [data[offset + 1 : offset + 1 + BLOCK_SIZE] for offset in offsets]
    )
    input_batch = input_batch.to(DEVICE)
    target_batch = target_batch.to(DEVICE)
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
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_model(train_data, val_data, model, optimizer, decode, iterations=20000)


if __name__ == "__main__":
    main()
