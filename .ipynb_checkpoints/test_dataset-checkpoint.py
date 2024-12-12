# test_dataset.py

from data import Im2LatexDataset

def test():
    test_files = [
        './data/test_0.pkl',
        # 必要に応じて他のファイルを追加
    ]
    dataset = Im2LatexDataset(test_files, max_len=64)
    print(f"Total samples: {len(dataset)}")
    if len(dataset) > 0:
        print(dataset[0])

if __name__ == "__main__":
    test()
