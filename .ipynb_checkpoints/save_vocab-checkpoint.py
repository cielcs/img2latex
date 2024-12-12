# save_vocab.py

from build_vocab import Vocab, save_vocab
import os

def create_vocab(data_dir, split):
    """
    語彙を作成し、保存する
    Args:
        data_dir (str): データディレクトリのパス
        split (str): データの分割（'train', 'validate', 'test'）
    Returns:
        Vocab: 作成されたVocabオブジェクト
    """
    formulas_file = os.path.join(data_dir, "im2latex_formulas.norm.lst")
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]
    
    vocab = Vocab(min_freq=1)  # 必要に応じてmin_freqを調整
    vocab.build_vocab(formulas)
    return vocab

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Save Vocab for Im2Latex")
    parser.add_argument("--data_path", type=str, default="./data/", help="The dataset's dir")
    args = parser.parse_args()

    vocab = create_vocab(args.data_path, split="train")  # 通常は訓練データから語彙を構築
    save_vocab(args.data_path, vocab)
    print("Vocab has been created and saved successfully.")
