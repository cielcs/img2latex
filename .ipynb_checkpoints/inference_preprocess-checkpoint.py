# preprocess_inference.py
import os
from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # バッチ次元を追加

def main():
    input_dir = 'my_images/'  # 変換したい画像が入っているディレクトリ
    output_dir = 'processed_images/'
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # モデルが期待するサイズに合わせる
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_dir, filename)
            image_tensor = preprocess_image(image_path, transform)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.pt')
            torch.save(image_tensor, output_path)
            print(f"Processed and saved {output_path}")

if __name__ == '__main__':
    main()
