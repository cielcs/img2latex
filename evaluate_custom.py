# evaluate_custom.py
from os.path import join
import argparse
import glob
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Im2LatexDataset
from build_vocab import Vocab, load_vocab
from utils import collate_fn
from model import LatexProducer, Im2LatexModel
from model.score import score_files

# カスタムデータセットのインポート
from custom_dataset import CustomImageDataset
from torchvision import transforms

def decode_predictions(predictions, vocab):
    """
    モデルの予測結果をLaTeXコードにデコードする関数。
    predictions: list of lists or list of tensors
    return: list of LaTeX strings
    """
    latex_outputs = []

    for seq in predictions:
        latex = ""
        for token in seq:
            # トークンがテンソルの場合は数値を取得
            token_id = token.item() if isinstance(token, torch.Tensor) else token
            # <end> トークンで終了
            if token_id == vocab.sign2id.get('<end>', 2):
                break
            # トークンIDをLaTeX文字列に変換
            sign = vocab.id2sign.get(token_id, '')
            latex += sign
            print(f"Decoded token_id {token_id} to '{sign}'")
        latex_outputs.append(latex)

    return latex_outputs

def main():

    parser = argparse.ArgumentParser(description="Im2Latex Custom Evaluating Program")
    parser.add_argument('--model_path', required=True,
                        help='path of the evaluated model')

    # 新しい引数を追加
    parser.add_argument("--input_dir", type=str,
                        default="./my_images/", help="The directory of input images")
    parser.add_argument("--output_dir", type=str,
                        default="./inference_results/", help="The directory to store results")
    parser.add_argument("--vocab_path", type=str,
                        required=True, help="The directory containing vocab.pkl")
    parser.add_argument("--cuda", action='store_true',
                        default=True, help="Use cuda or not")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for beam search")
    parser.add_argument("--max_len", type=int,
                        default=64, help="Max step of decoding")

    args = parser.parse_args()

    # モデルのロード
    print("Loading model...")
    checkpoint = torch.load(join(args.model_path), map_location='cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    model_args = checkpoint['args']

    # 語彙のロード
    print("Loading vocabulary...")
    vocab = load_vocab(args.vocab_path)  # 語彙のロード。vocab.pkl が存在するディレクトリを指定
    use_cuda = True if args.cuda and torch.cuda.is_available() else False

    device = torch.device("cuda" if use_cuda else "cpu")

    # カスタムデータセットの準備
    print("Constructing data loader...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    custom_dataset = CustomImageDataset(image_dir=args.input_dir, transform=transform)
    data_loader = DataLoader(
        custom_dataset,
        batch_size=1,  # 1つの画像ずつ処理
        shuffle=False,
        pin_memory=True if use_cuda else False,
        num_workers=1  # 警告を回避するために1に設定
    )

    # モデルの構築とロード
    print("Constructing model...")
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

    # 結果を保存するディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    # LatexProducerの設定
    latex_producer = LatexProducer(
        model, vocab, max_len=args.max_len,
        use_cuda=use_cuda, beam_size=args.beam_size
    )

    # 推論ループ
    print("Starting inference...")
    for images, img_names in tqdm(data_loader):
        try:
            # 推論
            results = latex_producer(images.to(device))
            print(f"Raw predictions for {img_names[0]}: {results}")
            # デコード
            latex = decode_predictions(results, vocab)
        except RuntimeError as e:
            print(f"RuntimeError encountered: {e}")
            continue  # エラーが発生した画像はスキップ

        # 結果の保存
        img_name = img_names[0]
        output_path = os.path.join(args.output_dir, os.path.splitext(img_name)[0] + '.txt')
        with open(output_path, 'w') as f:
            for l in latex:
                f.write(l + '\n')
        
        print(f"Processed {img_name} and saved LaTeX to {output_path}")

if __name__ == "__main__":
    main()
