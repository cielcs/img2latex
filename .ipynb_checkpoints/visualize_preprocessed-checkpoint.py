# visualize_preprocessed.py
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def show_tensor_image(tensor):
    """
    テンソルをPIL画像に変換して表示する関数。
    """
    tensor = tensor.squeeze(0)  # バッチ次元を削除
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)  # 正規化を元に戻す
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def main():
    tensor_path = './processed_images/im2tex01.pt'  # 確認したい画像のテンソルパスを指定
    tensor = torch.load(tensor_path)
    show_tensor_image(tensor)

if __name__ == '__main__':
    main()
