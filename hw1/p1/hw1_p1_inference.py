import numpy as np
import pandas as pd
import torch
import os, argparse
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm
import random
import torchvision.models as models

def get_args():
    parser = argparse.ArgumentParser(description='Evaluate test set with pretrained resnet50 model.')
    parser.add_argument('--input', '-t', type=str, default=None, help='test data')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output directory path')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model path')
    return parser.parse_args()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class ImageDataset(Dataset):

  def __init__(self, path, tfm=test_transform):
    super(ImageDataset).__init__()
    self.path = path
    self.img_filenames = [file for file in os.listdir(path) if file.endswith(".png")]
    self.img_filenames.sort()
    self.transform = tfm

  def __len__(self):
    return len(self.img_filenames)
  
  def __getitem__(self, idx):
    f_name = os.path.join(self.path, self.img_filenames[idx])
    img = Image.open(f_name)
    img = self.transform(img)
    
    return img

def create_output_directory(output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def inference(datapath, outpath, modelpath, batch_size=8):
    create_output_directory(outpath)
    inference_set = ImageDataset(datapath)
    inference_loader = DataLoader(
        inference_set, batch_size=batch_size, shuffle=False, pin_memory=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = models.resnet50(weights='DEFAULT')
    model = model.to(device)
    model.load_state_dict(torch.load(modelpath))
    # print("load model done")
    model.eval()

    result = []

    for inference_data in inference_loader:
        inference_data = inference_data.to(device)
        logits = model(inference_data)
        x = torch.argmax(logits, dim=-1).cpu().detach().numpy()
        result.append(x)
    result = np.concatenate(result)

    df = pd.DataFrame(
        {'filename': inference_set.img_filenames, 'label': result})
    df.to_csv(outpath, index=False)
if __name__ == "__main__":
    print("start")
    args = get_args()
    inference(args.input, args.output, args.model)
    print("finish")