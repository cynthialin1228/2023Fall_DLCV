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
    parser = argparse.ArgumentParser(description='Evaluate test set with BYOL.')
    parser.add_argument('--csv', '-c', type=str, default=None, help='input csv path')
    parser.add_argument('--output', '-o', type=str, default=None, help='output csv path')
    parser.add_argument('--input', '-t', type=str, default=None, help='test data')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model path')
    return parser.parse_args()

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class full_model(nn.Module):
    def __init__(self) -> None:
        super(full_model, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.fc = nn.Sequential(
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Linear(512, 250),
            nn.ReLU(),
            nn.Linear(250, 65)
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.fc(out)
        return out

class ImageDataset(Dataset):

  def __init__(self, in_csv, path, tfm=test_transform):
    super(ImageDataset).__init__()
    self.path = path
    gt = pd.read_csv(in_csv)
    self.files = gt['filename'].to_list()
    self.files.sort()
    self.transform = tfm

  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, idx):
    f_name = os.path.join(self.path, self.files[idx])
    img = Image.open(f_name)
    img = self.transform(img)
    
    return img, f_name, idx

def create_output_directory(output_path):
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def inference(in_csv, inpath, outpath, modelpath, batch_size=1):
    create_output_directory(outpath)
    inference_set = ImageDataset(in_csv, inpath)
    inference_loader = DataLoader(
        inference_set, batch_size=batch_size, shuffle=False, pin_memory=True)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model = full_model().to(device)
    state = torch.load(modelpath)
    model.load_state_dict(state['model_state_dict'])
    print("load model done")
    model.eval()
    ids = []
    files = []
    preds = []
    for images, filenames, idx in inference_loader:
        with torch.no_grad():
            logits = model(images.to(device))
        pred = logits.argmax(dim=-1)
        files.append(filenames[0].split("/")[-1])
        preds.append(pred[0].cpu().detach().numpy())
        ids.append(idx[0].numpy())
    # print(ids)
    # print(files)
    # print(preds)
    df = pd.DataFrame(
        { 'id': ids, 'filename': files, 'label': preds})
    df.to_csv(outpath, index=False)
if __name__ == "__main__":
    print("start")
    args = get_args()
    inference(args.csv, args.input, args.output, args.model)
    print("finish")