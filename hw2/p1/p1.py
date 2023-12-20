# -*- coding: utf-8 -*-
"""temp_dlcv_hw2_p1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZAKq_36c66LLUWkNSKjSPB9jF_kJYmEJ
"""

# !gdown 1XUl4tqq3kKyWRe2lyHZ8s7U9yc_x3mWT -O hw2_data.zip

# !unzip ./hw2_data.zip -d ./
# !rm hw2_data.zip

import numpy as np
import torch
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torchvision.transforms as transforms
from PIL import Image
from tqdm.auto import tqdm
import random

np.random.seed(10901041)
torch.manual_seed(10901041)

class DigitDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=transforms.ToTensor()):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        if split == 'all':
            self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
            self.labels = [-1 for _ in self.image_files]
        else:
            label_file = os.path.join(data_dir, f'{split}.csv')
            with open(label_file, 'r') as f:
                lines = f.readlines()
                self.labels = [int(line.strip().split(',')[1]) for line in lines[1:]]
                self.imgs = [line.strip().split(',')[0] for line in lines[1:]]
                valid_image_files = [img for img in self.imgs]
                self.image_files = sorted(
                    [
                        os.path.join(data_dir, "data", x)
                        for x in os.listdir(os.path.join(data_dir, "data"))
                        if x in valid_image_files
                    ]
                )


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

"""# Model"""

import math
from typing import List
# Ref: https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/unet.py
# Ref: https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_res: bool = False) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            # this adds on correct residual in case channels have increased
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2

class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            ResidualConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.model(x)

class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        return self.model(torch.cat((x, skip), 1))

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.input_dim = input_dim

    def forward(self, x):
        return self.model(x.view(-1, self.input_dim))

class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat = 256, n_classes=10):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, n_feat)
        self.up2 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        # noisy image, c is context label

        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask))
        c = c * context_mask

        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        # could concatenate the context embedding here instead of adaGN
        # hiddenvec = torch.cat((hiddenvec, temb1, cemb1), 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out

# Ref: https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py

def coefficients(start_beta, end_beta, T):
    beta_t = (end_beta - start_beta) * torch.arange(0, T + 1, dtype=torch.float32) / T + start_beta
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    bar_alpha_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()
    sqrt_bar_alpha = torch.sqrt(bar_alpha_t)
    sqrtmab = torch.sqrt(1 - bar_alpha_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,
        "beta_t": beta_t,
        "sqrt_beta_t": sqrt_beta_t,
        "oneover_sqrta": oneover_sqrta,
        "bar_alpha_t": bar_alpha_t,
        "sqrtab": sqrt_bar_alpha,
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(self, model, start_beta, end_beta, T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.model = model.to(device)
        self.T = T
        self.device = device
        self.drop_prob = drop_prob
        self.loss = nn.MSELoss()

        coeffs = coefficients(start_beta, end_beta, T)
        for k, v in coeffs.items():
            self.register_buffer(k, v)

    def forward(self, x, context):
        # Algorithm 1 by https://arxiv.org/pdf/2006.11239.pdf
        # Sample t uniformly for each batch element
        uniform_t = torch.randint(1, self.T + 1, (x.size(0),), device=self.device)
        epsilon = torch.randn_like(x)

        # Noisy version of the input data x
        x_noisy = (
            self.sqrtmab[uniform_t, None, None, None] * epsilon
            + self.sqrtab[uniform_t, None, None, None] * x
        )

        # Apply dropout to avoid overfitting
        context_mask = torch.bernoulli(torch.zeros_like(context)+ self.drop_prob).to(self.device)
        # Predict noise and return MSE loss
        predicted_noise = self.model(x_noisy, c, uniform_t / self.T, context_mask)
        return self.loss(predicted_noise, epsilon)

    @torch.no_grad()
    def sample(self, num_samples, size, guide_w=0.0, time_step=False):
        x_i = torch.randn(num_samples, *size, device=self.device)
        c_i = torch.arange(self.model.n_classes, device=self.device).repeat(num_samples // self.model.n_classes)

        # No context dropout at test time
        context_mask = torch.zeros_like(c_i, device=self.device)

        # Batch doubling for context-free guidance
        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[num_samples:] = 1

        x_i_store = []
        # Algorithm 2 by https://arxiv.org/pdf/2006.11239.pdf
        for i in reversed(range(self.T)):
            t_i = torch.full((num_samples,), i / self.T, device=self.device).repeat(2)
            z = torch.randn(num_samples, *size, device=self.device) if i > 0 else 0

            # Predict noise and apply guidance
            # Use guide_w to perform a weighted combination of the predicted noise epsilon for two different sets of conditions
            epsilon = self.model(x_i.repeat(2, 1, 1, 1), c_i, t_i, context_mask)
            epsilon = (1 + guide_w) * epsilon[:num_samples] - guide_w * epsilon[num_samples:]

            x_i = (self.oneover_sqrta[i] * (x_i - epsilon * self.mab_over_sqrtmab[i])+ self.sqrt_beta_t[i] * z)[:num_samples]

            if time_step and (i % 20 == 0 or i in {0, self.T - 1}):
                x_i_store.append(x_i[0].detach().cpu())

        return x_i, x_i_store

"""# Load data"""

mnistm_transform = transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4632, 0.4669, 0.4195], std=[0.2520, 0.2365, 0.2605])
    ])
mnistm_train_dataset = DigitDataset(data_dir="./hw2_data/digits/mnistm", split="train", transform=mnistm_transform)
mnistm_train_dataloader = DataLoader(mnistm_train_dataset, batch_size=64, shuffle=True)
# mnistm_val_dataset = DigitDataset(data_dir="./hw2_data/digits/mnistm", split="val", transform=mnistm_transform)
# mnistm_val_dataloader = DataLoader(mnistm_val_dataset, batch_size=64, shuffle=True)

"""# Train"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

num_epochs = 150
T = 500
ckpt_dir = './'
img_size=28,
n_feat=256,
n_classes=10,
lr=1e-4
guide_weights = [0.0, 0.5, 2.0]

model = ContextUnet(in_channels=3)
model = DDPM(model = model, start_beta=1e-4, end_beta = 0.02, T=T, device = device, drop_prob = 0.1)
model = model.float().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 - epoch/num_epochs)

best_acc = 0
batch_size=256
for epoch in range(num_epochs):
    train_loss = []
    model.train()
    for imgs, c in tqdm(mnistm_train_dataloader):
        optimizer.zero_grad()
        # Forward pass and compute loss
        imgs = imgs.float().to(device)
        c = c.to(device)
        loss = model(imgs, c)
        loss.backward()  # Backward pass
        train_loss.append(loss.item())  # Record loss
        optimizer.step()  # Update weights

    train_loss_sum = np.sum(train_loss)
    # Make sure that batch_size is a scalar
    assert np.isscalar(batch_size), "batch_size should be a scalar value"
    # Calculate the average loss
    average_train_loss = train_loss_sum / batch_size
    print(f"epoch {epoch} has loss: {average_train_loss}")

    model.eval()
    with torch.no_grad():
        for w in guide_weights:# Iterate over guide weights for generative guidance
            # Generate samples with the current guide weight
            generate_sample, _ = model.sample(num_samples=10, size = (3, 28, 28), guide_w=w)
            # Perform accuracy check and update avg_acc accordingly.
            gen_samples = []

            for idx in range(generate_sample.size(0)):
                img = generate_sample[idx, :, :, :]
                gen_samples.append(img)
        scheduler.step()

        if (epoch+1) % 30 == 0:
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'epoch': epoch},
                    os.path.join(ckpt_dir, f"{epoch}.pt"))
            print(f"epoch {epoch}: save model")

"""# Plot"""

from torchvision.utils import save_image
random.seed(10901041)
torch.manual_seed(10901041)

with torch.no_grad():
    x_i, x_i_store = model.sample(100, size=(
        3, 28, 28), guide_w=2, time_step=True)
    x_i = x_i.reshape(10, 10, 3, 28, 28)
    x_i = torch.transpose(x_i, 0, 1)
    x_i = x_i.reshape(-1, 3, 28, 28)
    save_image(x_i, './100_samples.png', nrow=10)

indices = [0, 6, 11, 16, 21, 25]
selected_samples = torch.stack([x_i_store[i] for i in indices])
selected_samples = selected_samples.reshape(6, 3, 28, 28)
save_image(selected_samples, './selected_samples.png')