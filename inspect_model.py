# inspect_model.py
import torch
from model import Im2LatexModel
from build_vocab import Vocab, load_vocab
import argparse

def main():
    parser = argparse.ArgumentParser(description="Inspect Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to the vocab directory containing vocab.pkl")
    parser.add_argument("--cuda", action='store_true', default=True, help="Use cuda or not")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']

    # Load vocab
    vocab = load_vocab(args.vocab_path)

    # Initialize model
    model = Im2LatexModel(
        out_size=len(vocab),
        emb_size=model_args.emb_dim,
        dec_rnn_h=model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Print model summary
    print(model)

    # Print number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

if __name__ == "__main__":
    main()
