# evaluate.py

from os.path import join
from functools import partial
import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from data import Im2LatexDataset
from build_vocab import Vocab, load_vocab
from utils import collate_fn
from model import LatexProducer, Im2LatexModel
from model.score import score_files


def main():

    parser = argparse.ArgumentParser(description="Im2Latex Evaluating Program")
    parser.add_argument('--model_path', required=True,
                        help='path of the evaluated model')

    # model args
    parser.add_argument("--data_path", type=str,
                        default="./sample_data/", help="The dataset's dir")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=32)
    parser.add_argument("--result_path", type=str,
                        default="./results/result.txt", help="The file to store result")
    parser.add_argument("--ref_path", type=str,
                        default="./results/ref.txt", help="The file to store reference")
    parser.add_argument("--max_len", type=int,
                        default=64, help="Max step of decoding")
    parser.add_argument("--split", type=str,
                        default="validate", help="The data split to decode")
    
    # 新しい引数を追加
    parser.add_argument("--max_test_files", type=int, default=None,
                        help="Maximum number of test files to load")

    args = parser.parse_args()

    # セキュリティ警告への対応として weights_only=True を設定（必要に応じて）
    # ただし、データが安全であることを確認してください。
    if torch.__version__ >= '1.13.0':  # 確認: weights_onlyはPyTorch 1.13以降で利用可能
        checkpoint = torch.load(join(args.model_path))
    else:
        checkpoint = torch.load(join(args.model_path))

    model_args = checkpoint['args']

    # 词典のロードと関連パラメータの設定
    vocab = load_vocab(args.data_path)
    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    device = torch.device("cuda" if use_cuda else "cpu")

    # テストデータファイルの取得
    print("Construct data loader...")
    test_files = glob.glob(os.path.join(args.data_path, f'{args.split}_*.pkl'))
    if not test_files:
        raise FileNotFoundError(f"No {args.split} files found in {args.data_path} with pattern '{args.split}_*.pkl'")

    # 最大ファイル数を適用
    if args.max_test_files is not None:
        test_files = test_files[:args.max_test_files]
        print(f"Using {len(test_files)} {args.split} files")

    # 複数のデータセットを結合
    test_dataset = Im2LatexDataset(test_files, args.max_len)
    
    # DataLoaderの作成
    data_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, vocab.sign2id),
        pin_memory=True if use_cuda else False,
        num_workers=4
    )

    # モデルの構築とロード
    print("Construct model")
    model = Im2LatexModel(
        len(vocab), model_args.emb_dim, model_args.dec_rnn_h,
        add_pos_feat=model_args.add_position_features,
        dropout=model_args.dropout
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 結果と参照のファイルを開く
    os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.ref_path), exist_ok=True)
    result_file = open(args.result_path, 'w')
    ref_file = open(args.ref_path, 'w')

    # LatexProducerの設定
    latex_producer = LatexProducer(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size
    )

    # 評価ループ
    for imgs, tgt4training, tgt4cal_loss in tqdm(data_loader):
        try:
            reference = latex_producer._idx2formulas(tgt4cal_loss)
            results = latex_producer(imgs)
        except RuntimeError as e:
            print(f"RuntimeError encountered: {e}")
            break

        result_file.write('\n'.join(results) + '\n')
        ref_file.write('\n'.join(reference) + '\n')

    result_file.close()
    ref_file.close()

    # スコアの計算
    score = score_files(args.result_path, args.ref_path)
    print("beam search result:", score)


if __name__ == "__main__":
    main()
