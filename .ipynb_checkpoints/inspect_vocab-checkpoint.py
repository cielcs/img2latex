# inspect_vocab.py
from build_vocab import Vocab, load_vocab
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inspect Vocab")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab directory containing vocab.pkl")
    args = parser.parse_args()

    vocab = load_vocab(args.vocab_path)
    print(f"Vocabulary size: {len(vocab)}")
    print("Sample mappings:")
    for idx, sign in list(vocab.id2sign.items())[:300]:
        print(f"{idx}: {sign}")

if __name__ == "__main__":
    main()
