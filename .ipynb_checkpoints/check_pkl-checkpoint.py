import pickle

def check_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        print(f"Type of data: {type(data)}")
        if isinstance(data, list):
            print(f"Number of samples: {len(data)}")
            if len(data) > 0:
                print(f"Sample 0: {data[0]}")
        else:
            print("Data is not a list.")

if __name__ == "__main__":
    check_pkl('./data/test_0.pkl')
