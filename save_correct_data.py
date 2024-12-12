# save_correct_data.py

from os.path import join
import argparse

from PIL import Image
from torchvision import transforms
import torch
import time
import os

def log_gpu_memory(interval=10):
    import threading
    import torch
    import time

    def log():
        while True:
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                print(f"GPU Memory Allocated: {memory_allocated / 1024 ** 2:.2f} MB")
                print(f"GPU Memory Reserved: {memory_reserved / 1024 ** 2:.2f} MB")
            else:
                print("CUDA is not available.")
            time.sleep(interval)
    
    thread = threading.Thread(target=log, args=(), daemon=True)
    thread.start()

def preprocess(data_dir, split):
    assert split in ["train", "validate", "test"]

    print("Process {} dataset...".format(split))
    images_dir = join(data_dir, "formula_images_processed")

    formulas_file = join(data_dir, "im2latex_formulas.norm.lst")
    with open(formulas_file, 'r') as f:
        formulas = [formula.strip('\n') for formula in f.readlines()]

    split_file = join(data_dir, "im2latex_{}_filter.lst".format(split))
    transform = transforms.ToTensor()
    batch = []  # バッチを保持する一時的なリスト
    batch_size = 1000
    batch_index = 0

    with open(split_file, 'r') as f:
        for line in f:
            img_name, formula_id = line.strip('\n').split()
            # 画像と数式を読み込む
            img_path = join(images_dir, img_name)
            if not os.path.isfile(img_path):
                print(f"Image not found: {img_path}")
                continue
            try:
                img = Image.open(img_path).convert('RGB')  # RGBに変換
                img_tensor = transform(img)
                formula = formulas[int(formula_id)]
                pair = (img_tensor, formula)
                batch.append(pair)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

            # バッチが満たされたら保存
            if len(batch) >= batch_size:
                out_file = join(data_dir, "{}_{}.pkl".format(split, batch_index))
                torch.save(batch, out_file)
                print("Saved batch {} to {}".format(batch_index, out_file))
                batch = []  # バッチをリセット
                batch_index += 1

        # 最後に残ったデータを保存
        if len(batch) > 0:
            out_file = join(data_dir, "{}_{}.pkl".format(split, batch_index))
            torch.save(batch, out_file)
            print("Saved batch {} to {}".format(batch_index, out_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Im2Latex Data Preprocess Program")
    parser.add_argument("--data_path", type=str,
                        default="./data/", help="The dataset's dir")
    args = parser.parse_args()

    # GPUメモリのログを開始（オプション）
    log_gpu_memory(interval=10)
    
    splits = ["validate", "test", "train"]
    for s in splits:
        preprocess(args.data_path, s)
