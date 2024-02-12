import argparse


def generate_encoder_decoder(token_list):
    token_to_encoding = {c: i for i, c in enumerate(token_list)}
    encoding_to_token = {i: c for i, c in enumerate(token_list)}

    encode = lambda tokens: list(map(lambda t: token_to_encoding[t], tokens))
    decode = lambda encodings: list(map(lambda e: encoding_to_token[e], encodings))

    return encode, decode


def extract_token_list(text):
    return sorted(list(set(text)))


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
    token_list = extract_token_list(content)
    encode, decode = generate_encoder_decoder(token_list)
    print(encode("Hello world"))
    print("".join(decode(encode("Hello world"))))


if __name__ == "__main__":
    main()
