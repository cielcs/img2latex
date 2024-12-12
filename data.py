# data.py

from torch.utils.data import Dataset
import torch
import pickle
import os

class Im2LatexDataset(Dataset):
    def __init__(self, file_paths, max_len):
        """
        Args:
            file_paths (list of str): List of paths to the .pkl files.
            max_len (int): Maximum length of the formula.
        """
        self.file_paths = file_paths
        self.max_len = max_len
        self.pairs = []
        
        # 各ファイルのデータをロードして結合
        for file_path in self.file_paths:
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                continue
            with open(file_path, 'rb') as f:
                try:
                    data = torch.load(f)  # torch.loadを使用
                    if isinstance(data, list):
                        self.pairs.extend(data)
                        print(f"Loaded {len(data)} samples from {file_path}")
                    else:
                        raise ValueError(f"Expected list in {file_path}, but got {type(data)}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sample = self.pairs[idx]

        # 必要に応じて前処理（例：max_len の適用）
        if len(sample[1]) > self.max_len:  # sampleは (image_tensor, formula) のタプル
            sample = (sample[0], sample[1][:self.max_len])

        return sample
