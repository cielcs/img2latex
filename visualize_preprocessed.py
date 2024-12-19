# visualize_preprocessed.py
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os

def show_and_save_tensor_image(tensor, output_path):
    """
    テンソルをPIL画像に変換して保存する関数。
    
    Args:
        tensor (torch.Tensor): 前処理された画像テンソル
        output_path (str): 保存する画像ファイルのパス
    """
    tensor = tensor.squeeze(0)  # バッチ次元を削除
    # 正規化を元に戻す
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1) + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    tensor = torch.clamp(tensor, 0, 1)
    image = transforms.ToPILImage()(tensor)
    
    # 画像を保存
    image.save(output_path)
    print(f"Preprocessed image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize and Save Preprocessed Image Tensor")
    parser.add_argument(
        '--tensor_path',
        type=str,
        required=True,
        help='Path to the preprocessed image tensor file (e.g., ./processed_images/image1.pt)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save the output image file (e.g., ./preprocessed_images/image1_preprocessed.png). If not provided, it will be derived from tensor_path.'
    )
    
    args = parser.parse_args()
    
    tensor_path = args.tensor_path
    
    # 出力パスが指定されていない場合、デフォルトで同じディレクトリに保存
    if args.output_path is None:
        base_name = os.path.splitext(os.path.basename(tensor_path))[0]
        output_path = os.path.join(os.path.dirname(tensor_path), f"{base_name}_preprocessed.png")
    else:
        output_path = args.output_path
    
    # テンソルのロード
    tensor = torch.load(tensor_path)
    
    # 画像の保存
    show_and_save_tensor_image(tensor, output_path)

if __name__ == '__main__':
    main()
