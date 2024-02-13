import torch
import time
import signal
import sys
import os
import argparse
from functools import wraps
from bigram import BigramLanguageModel
from transformer import TransformerLanguageModel


MODEL = None
CURRENT_ITERATION = 0
OPTIMIZER = None


TRAINING_DATA_PERCENTAGE = 0.9
BLOCK_SIZE = 256
BATCH_SIZE = 64
NUM_HEADS = 6
EMBEDDING_DIMENSIONS = NUM_HEADS * 64
NUM_BLOCKS = 6
# EMBEDDING_DIMENSIONS = NUM_HEADS * 2
# NUM_BLOCKS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVALUATION_ITERATIONS = 200
PRINT_ITERATIONS = 5
TOTAL_ITERATIONS = 5000
LEARNING_RATE = 3e-4
DROPOUT = 0.2


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


def print_loss(train_data, val_data, model):
    global CURRENT_ITERATION
    train_loss = estimate_loss(train_data, model)
    val_loss = estimate_loss(val_data, model)
    print(f"iteration: {CURRENT_ITERATION}, training: {train_loss}, val loss: {val_loss}")


def print_sample_output(model, encode, decode):
    model.eval()
    generated = model.generate(
        torch.tensor(encode(" "), dtype=torch.long, device=DEVICE)[None, :],
        num_new_tokens=100,
    )
    model.train()
    print("sample output:")
    print(decode(generated[0].tolist()))
    print()


@timing_decorator
def train_model(train_data, val_data, model, optimizer, encode, decode, iterations):
    global CURRENT_ITERATION
    for i in range(iterations + 1):
        inputs, targets = sample_batch(train_data)
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % PRINT_ITERATIONS == 0:
            print_loss(train_data, val_data, model)
            #print_sample_output(model, encode, decode)
        CURRENT_ITERATION += 1


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
    parser.add_argument("-s", "--statefile")
    parser.add_argument("filename")
    return parser.parse_args()


def save_checkpoint(filepath):
    global CURRENT_ITERATION
    global MODEL
    global OPTIMIZER
    if MODEL is not None and OPTIMIZER is not None:
        torch.save({
            "iteration": CURRENT_ITERATION,
            "model_state": MODEL.state_dict(),
            "optimizer_state": OPTIMIZER.state_dict(),
        }, filepath)


def load_checkpoint(filepath):
    global CURRENT_ITERATION
    global MODEL
    global OPTIMIZER
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        MODEL.load_state_dict(checkpoint["model_state"])
        OPTIMIZER.load_state_dict(checkpoint["optimizer_state"])
        CURRENT_ITERATION = checkpoint["iteration"]
        print(f"Loaded state file from {filepath}, iteration: {CURRENT_ITERATION}")


def main():
    global CURRENT_ITERATION
    global MODEL
    global OPTIMIZER

    args = parse_args()
    def signal_handler(*_):
        save_checkpoint(args.statefile)
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    content = read_file(args.filename)
    token_universe = extract_token_universe(content)
    encode, decode = generate_encoder_decoder(token_universe)
    data = torch.tensor(encode(content), dtype=torch.long)
    train_data, val_data = split_data(data, TRAINING_DATA_PERCENTAGE)
    MODEL = TransformerLanguageModel(
        max_tokens=len(token_universe),
        max_block_size=BLOCK_SIZE,
        embedding_dimensions=EMBEDDING_DIMENSIONS,
        num_heads=NUM_HEADS,
        num_transformer_blocks=NUM_BLOCKS,
        dropout=DROPOUT,
    )
    MODEL = MODEL.to(DEVICE)
    OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), lr=LEARNING_RATE)
    load_checkpoint(args.statefile)
    train_model(
        train_data,
        val_data,
        MODEL,
        OPTIMIZER,
        encode,
        decode,
        iterations=TOTAL_ITERATIONS,
    )


if __name__ == "__main__":
    main()
