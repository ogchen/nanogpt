import argparse
import train
import torch
import os
from transformer import TransformerLanguageModel


USERS = ["Oscar", "Chandan"]


def parse_args():
    parser = argparse.ArgumentParser(
        prog="nanogpt", description="Executes a GPT language model"
    )
    parser.add_argument("-s", "--statefile")
    parser.add_argument("-t", "--tokens", type=int)
    parser.add_argument("inputs")
    return parser.parse_args()


def get_user_from_stdin():
    user = None
    prompt = f"Select a user from {USERS}: "
    while user not in USERS:
        user = input(prompt)
        prompt = f"Invalid user '{user}' selected, please select one from {USERS}: "
    return user


def get_message(user_prompt):
    prompt = f"{user_prompt}"
    return input(prompt)


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
        user = get_user_from_stdin()
        responder = [u for u in USERS if u != user][0]
        user_prompt = f"{user}: "
        responder_prompt = f"{responder}: "
        input_message = f"{user_prompt}{get_message(user_prompt)}\n{responder_prompt}"
        print(responder_prompt, end="")

        while True:
            input = torch.tensor(encode(input_message), dtype=torch.long)[None, :]
            next_token = model.get_next_token(input)
            input = torch.cat((input, next_token), dim=1)[:, -train.BLOCK_SIZE :]
            input_message += decode(next_token[0].tolist())
            input_message = input_message[-train.BLOCK_SIZE :]
            print(decode(next_token[0].tolist()), end="")

            if input_message[-len(user_prompt) :] == user_prompt:
                input_message += f"{get_message('')}\n{responder}: "
                input_message = input_message[-train.BLOCK_SIZE :]
                print(responder_prompt, end="")


if __name__ == "__main__":
    main()
