import argparse

def read_file(filename):
    with open(filename, mode="r") as f:
        text = f.read()
    return text

def parse_args():
    parser = argparse.ArgumentParser(
        prog="nanogpt",
        description="Executes a GPT language model"
    )
    parser.add_argument("filename")
    return parser.parse_args()

def main():
    args = parse_args()
    content = read_file(args.filename)
    print(content[:1000])

if __name__ == "__main__":
    main()