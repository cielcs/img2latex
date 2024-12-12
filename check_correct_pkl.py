# check_correct_pkl.py

import torch

def check_correct_pkl(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = torch.load(f)
            print(f"Type of data: {type(data)}")
            print(f"Number of samples: {len(data)}")
            if len(data) > 0:
                sample = data[0]
                print(f"Sample 0 - Image Tensor Shape: {sample[0].shape}, Formula: {sample[1]}")
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")

if __name__ == "__main__":
    check_correct_pkl('./data/test_0.pkl')  # 正しく保存されたファイルパスに変更
