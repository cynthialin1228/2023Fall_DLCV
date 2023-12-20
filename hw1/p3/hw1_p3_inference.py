import numpy as np
import pandas as pd
import torch
import os, argparse
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate test set with pretrained resnet50 model.')
    parser.add_argument('--input', '-t', type=str, default=None, help='test data')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory path')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model path')
    return parser.parse_args()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class ImageDataset(Dataset):
  def __init__(self, path, tfm=test_transform):
    super(ImageDataset).__init__()
    self.path = path
    self.img_filenames = [file for file in os.listdir(path) if file.endswith(".jpg")]
    self.img_filenames.sort()
    self.transform = tfm
  def __len__(self):
    return len(self.img_filenames)
  def __getitem__(self, idx):
    f_name = os.path.join(self.path, self.img_filenames[idx])
    img = Image.open(f_name)
    img = self.transform(img)
    # img = torch.squeeze(img)
    return img, self.img_filenames[idx]
  
def safe_mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def inference(datapath, outpath, modelpath, batch_size=8):
    files = [f for f in sorted(os.listdir(datapath)) if f.endswith(".jpg")]
    inference_set = ImageDataset(datapath)
    inference_loader = DataLoader(
        inference_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = models.segmentation.deeplabv3_resnet50(num_classes=7)
    model = model.to(device)
    model.load_state_dict(torch.load(modelpath))
    # print("load model done")
    model.eval()
    pred_masks_cpu = []
    # result = []

    safe_mkdir(outpath)
    # colors = np.array([[0, 0, 0],
    #     [255, 255, 255],  
    #     [0, 0, 255],  
    #     [0, 255, 0],  # Class 3 color
    #     [255, 0, 255],    # Class 4 color
    #     [255, 255, 0],  # Class 5 color
    #     [0, 255, 255]       # Class 6 color
    # ])
    colors = np.array([
        [0, 255, 255],  # Class 0 color
        [255, 255, 0],  # Class 1 color
        [255, 0, 255],  # Class 2 color
        [0, 255, 0],    # Class 3 color
        [0, 0, 255],    # Class 4 color
        [255, 255, 255],  # Class 5 color
        [0, 0, 0]       # Class 6 color
    ])
    for image, filename in inference_loader:
        image, filename = image.to(device), filename
        with torch.no_grad():
            logits = model(image.to(device))
        logits = logits['out']
        pred = logits.argmax(dim=1)
        pred_masks_cpu.append(pred.cpu())
    pred_masks_numpy = torch.cat(pred_masks_cpu).numpy()
    output_images = np.zeros((pred_masks_numpy.shape[0], 512, 512, 3), dtype='uint8')
    print(pred_masks_numpy.shape)
    for idx in range(pred_masks_numpy.shape[0]):
        for height in range(512):
            for width in range(512):
                output_images[idx, height, width, :] = colors[pred_masks_numpy[idx, height, width]]
        image_filename = files[idx].replace('sat.jpg', 'mask.png')
        image_path = os.path.join(outpath, image_filename)
        output_image = Image.fromarray(output_images[idx])
        output_image.save(image_path)
       
if __name__ == "__main__":
    print("start")
    args = get_args()
    inference(args.input, args.output, args.model)
    print("finish")