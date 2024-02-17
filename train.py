import argparse
import json
import os
import signal
import sys
import time
import torch
from bigram import BigramLanguageModel
from functools import wraps
from src.transformer import TransformerLanguageModel

MODEL = None
EPOCH = 0
OPTIMIZER = None


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
def estimate_loss(data, model, config):
    model.eval()
    losses = torch.zeros(config["evaluation_iterations"])
    for k in range(config["evaluation_iterations"]):
        inputs, targets = sample_batch(data, config)
        _, loss = model(inputs, targets)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def print_loss(train_data, val_data, model, config):
    global EPOCH
    train_loss = estimate_loss(train_data, model, config)
    val_loss = estimate_loss(val_data, model, config)
    print(f"iteration: {EPOCH}, training: {train_loss}, val loss: {val_loss}")


def print_sample_output(model, encode, decode, config):
    model.eval()
    input = torch.tensor(encode(" "), dtype=torch.long, device=config["device"])[
        None, :
    ]
    generated = model.generate(
        input,
        num_new_tokens=100,
    )

    model.train()
    print("sample output:")
    print(decode(generated[0].tolist()))
    print()


@timing_decorator
def train_model(
    train_data, val_data, model, optimizer, encode, decode, statefile, config
):
    global EPOCH

    model.train()
    for i in range(EPOCH, config["total_iterations"] + 1):
        inputs, targets = sample_batch(train_data, config)
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % config["print_iterations"] == 0:
            print_loss(train_data, val_data, model, config)
            print_sample_output(model, encode, decode, config)
        if i % config["save_iterations"] == 0:
            save_checkpoint(statefile)
        EPOCH += 1


def sample_batch(data, config):
    assert len(data) >= config["block_size"] + 1
    offsets = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
    input_batch = torch.stack(
        [data[offset : offset + config["block_size"]] for offset in offsets]
    )
    target_batch = torch.stack(
        [data[offset + 1 : offset + 1 + config["block_size"]] for offset in offsets]
    )
    input_batch = input_batch.to(config["device"])
    target_batch = target_batch.to(config["device"])
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
        prog="train", description="Trains a GPT language model"
    )
    parser.add_argument("-c", "--config")
    parser.add_argument("-s", "--statefile")
    parser.add_argument("filename")
    return parser.parse_args()


def save_checkpoint(filepath):
    global EPOCH
    global MODEL
    global OPTIMIZER
    if MODEL is not None and OPTIMIZER is not None:
        torch.save(
            {
                "iteration": EPOCH,
                "model_state": MODEL.state_dict(),
                "optimizer_state": OPTIMIZER.state_dict(),
            },
            filepath,
        )


def load_checkpoint(filepath):
    global EPOCH
    global MODEL
    global OPTIMIZER
    if os.path.isfile(filepath):
        checkpoint = torch.load(filepath)
        MODEL.load_state_dict(checkpoint["model_state"])
        OPTIMIZER.load_state_dict(checkpoint["optimizer_state"])
        EPOCH = checkpoint["iteration"]
        print(f"Loaded state file from {filepath}, iteration: {EPOCH}")


def load_config(filepath):
    with open(filepath, mode="r") as f:
        config = json.load(f)
    if config["device"] == "cuda":
        assert torch.cuda.is_available()
    return config


def main():
    global EPOCH
    global MODEL
    global OPTIMIZER

    args = parse_args()

    def signal_handler(*_):
        save_checkpoint(args.statefile)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    config = load_config(args.config)
    content = read_file(args.filename)
    token_universe = extract_token_universe(content)
    encode, decode = generate_encoder_decoder(token_universe)
    data = torch.tensor(encode(content), dtype=torch.long)
    train_data, val_data = split_data(data, config["training_data_percentage"])
    MODEL = TransformerLanguageModel(
        max_tokens=len(token_universe),
        max_block_size=config["block_size"],
        embedding_dimensions=config["embedding_dimensions"],
        num_heads=config["num_heads"],
        num_transformer_blocks=config["num_blocks"],
        dropout=config["dropout"],
    )
    MODEL = MODEL.to(config["device"])
    OPTIMIZER = torch.optim.AdamW(MODEL.parameters(), lr=config["learning_rate"])
    load_checkpoint(args.statefile)
    train_model(
        train_data,
        val_data,
        MODEL,
        OPTIMIZER,
        encode,
        decode,
        args.statefile,
        config=config,
    )


if __name__ == "__main__":
    main()
