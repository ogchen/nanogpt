import argparse
import json
import os
import signal
import sys
import torch
from src.checkpoint import CheckpointManager
from src.tokenizer import Tokenizer
from src.transformer import TransformerLanguageModel
from src import utils

EPOCH = 0


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


def print_loss(epoch, train_data, val_data, model, config):
    train_loss = estimate_loss(train_data, model, config)
    val_loss = estimate_loss(val_data, model, config)
    print(f"iteration: {epoch}, training: {train_loss}, val loss: {val_loss}")


def print_sample_output(model, tokenizer, config):
    model.eval()
    input = torch.tensor(
        tokenizer.encode(" "), dtype=torch.long, device=config["device"]
    )[None, :]
    generated = model.generate(
        input,
        num_new_tokens=100,
    )

    model.train()
    print("sample output:")
    print(tokenizer.decode(generated[0].tolist()))
    print()


@utils.timing_decorator
def train_model(data, model, optimizer, tokenizer, checkpoint_manager, writer, config):
    train_data, val_data = split_data(data, config["training_data_percentage"])
    model.train()
    for i in range(checkpoint_manager.epoch, config["total_iterations"] + 1):
        checkpoint_manager.epoch = i
        inputs, targets = sample_batch(train_data, config)
        _, loss = model(inputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % config["print_iterations"] == 0:
            print_loss(i, train_data, val_data, model, config)
            print_sample_output(model, tokenizer, config)
        if i % config["save_iterations"] == 0:
            checkpoint_manager.save()


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


def read_file(filename):
    with open(filename, mode="r") as f:
        return f.read()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="train", description="Trains a GPT language model"
    )
    parser.add_argument("-c", "--config")
    parser.add_argument("-s", "--statefile")
    parser.add_argument("filename")
    return parser.parse_args()


def load_config(filepath):
    with open(filepath, mode="r") as f:
        config = json.load(f)
    if config["device"] == "cuda":
        assert torch.cuda.is_available()
    return config


def register_sigint_handler(checkpoint_manager):
    def signal_handler(*_):
        print(f"SIGINT received, saving model and exiting")
        checkpoint_manager.save()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


def main():
    args = parse_args()
    config = load_config(args.config)

    content = read_file(args.filename)
    tokenizer = Tokenizer(content)

    model = TransformerLanguageModel(
        max_tokens=tokenizer.count(),
        max_block_size=config["block_size"],
        dim_embedding=config["dim_embedding"],
        num_heads=config["num_heads"],
        num_transformer_blocks=config["num_blocks"],
        dropout=config["dropout"],
    )
    model = model.to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    checkpoint_manager = CheckpointManager(model, optimizer, args.statefile)
    checkpoint_manager.load()
    register_sigint_handler(checkpoint_manager)

    data = torch.tensor(tokenizer.encode(content), dtype=torch.long)
    train_model(
        data,
        model,
        optimizer,
        tokenizer,
        checkpoint_manager,
        config=config,
    )


if __name__ == "__main__":
    main()
