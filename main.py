import os 
import torch
import cv2
from RealESRGAN import RealESRGAN
from tqdm import tqdm

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    files = os.listdir("inputs")
    for i, image in tqdm(enumerate(files), total=len(files)):
        sr_image = cv2.imread(f"inputs/{image}")
        if sr_image is None:
            continue
        sr_image = model.predict(sr_image)
        cv2.imwrite(f'results/{i}.png', sr_image)