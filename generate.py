import argparse
import train
import torch
import os
from src.transformer import TransformerLanguageModel


def parse_args():
    parser = argparse.ArgumentParser(
        prog="nanogpt", description="Executes a GPT language model"
    )
    parser.add_argument("-s", "--statefile")
    parser.add_argument("-t", "--tokens", type=int)
    parser.add_argument("inputs")
    return parser.parse_args()


def main():
    args = parse_args()
    content = train.read_file(args.inputs)
    token_universe = train.extract_token_universe(content)
    encode, decode = train.generate_encoder_decoder(token_universe)
    model = TransformerLanguageModel(
        max_tokens=len(token_universe),
        max_block_size=train.BLOCK_SIZE,
        embedding_dimensions=train.EMBEDDING_DIMENSIONS,
        num_heads=train.NUM_HEADS,
        num_transformer_blocks=train.NUM_BLOCKS,
        dropout=train.DROPOUT,
    )
    if os.path.isfile(args.statefile):
        checkpoint = torch.load(args.statefile)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded model weights from {args.statefile}")
        model.eval()
        input = torch.tensor(encode(" "), dtype=torch.long)[None, :]
        while True:
            next_token = model.get_next_token(input)
            input = torch.cat((input, next_token), dim=1)[:, -train.BLOCK_SIZE :]
            print(decode(next_token[0].tolist()), end="")


if __name__ == "__main__":
    main()
