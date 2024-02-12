import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        prog="nanogpt",
        description="Executes a GPT language model"
    )
    parser.add_argument("filename")
    return parser.parse_args()

def main():
    args = parse_args()
    print(args)

if __name__ == "__main__":
    main()