# -*- coding: utf-8 -*-
"""dlcv_hw4

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1hAdrh44lNzwU-eYVbEMSIztq_POgwS0m
"""

# !pip install -r requirements.txt --no-cache-dir

# !gdown 1hF4z9U-xaoV4qaq9DbhTP-KKJTlwOUv_

# !unzip ./dataset.zip

# !unzip ./val_dataset.zip

"""Reference: https://github.com/kwea123/nerf_pl"""

# !git clone --recursive https://github.com/kwea123/nerf_pl

# !pip install -r /content/nerf_pl/requirements.txt --no-cache-dir

# !python nerf_pl/train.py \
#    --dataset_name blender \
#    --root_dir "./dataset" \
#    --N_importance 64 --img_wh 256 256 --noise_std 0 \
#    --num_epochs 3 --batch_size 1024 \
#    --optimizer ranger --lr 5e-4\
#    --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
#    --exp_name ""

# !python nerf_pl/train.py \
#    --dataset_name blender \
#    --root_dir "./dataset" \
#    --N_importance 64 --img_wh 256 256 --noise_std 0 \
#    --num_epochs 10 --batch_size 1024 \
#    --optimizer adam --lr 5e-4\
#    --lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 \
#    --exp_name ""

"""# test"""

# %cd nerf_pl

import torch
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import os
from models.rendering import *
from models.nerf import *

import metrics

from datasets import dataset_dict
from datasets.llff import *
from train import KlevrDataset

torch.backends.cudnn.benchmark = True
@torch.no_grad()
def f(rays):
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i+chunk],
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        dataset.white_back,
                        test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results

"""## visualize predict and depth image"""

torch.backends.cudnn.benchmark = True
output_folder = "../output_image"
os.makedirs(output_folder, exist_ok=True)
img_wh = (256, 256)
dataset = KlevrDataset(root_dir='../dataset', split='val')

embedding_xyz = Embedding(3, 10)
embedding_dir = Embedding(3, 4)

nerf_coarse = NeRF()
nerf_fine = NeRF()
ckpt_path = '../nerf_pl_model.ckpt'

load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

nerf_coarse.cuda().eval()
nerf_fine.cuda().eval();

models = [nerf_coarse, nerf_fine]
embeddings = [embedding_xyz, embedding_dir]

N_samples = 64
N_importance = 64
use_disp = False
chunk = 1024*32*4

for idx, image_id in enumerate(dataset.split_ids):
    # Construct the input image filename
    input_filename = f'{image_id:05d}'

    # Load and process the sample from the dataset
    sample = dataset[idx]
    rays = sample['rays'].cuda()
    results = f(rays)

    # Process the results (example with img_pred and depth_pred)
    img_pred = results['rgb_fine'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
    depth_pred = results['depth_fine'].view(img_wh[1], img_wh[0])

    plt.figure()
    plt.imshow(img_pred)
    plt.title(f'Predicted Image {input_filename}')
    plt.savefig(os.path.join(output_folder, f'{input_filename}_pred.png'))
    plt.close()

    # Save depth image
    plt.figure()
    plt.imshow(visualize_depth(depth_pred).permute(1, 2, 0))
    plt.title(f'Depth Image {input_filename}')
    plt.savefig(os.path.join(output_folder, f'{input_filename}_depth.png'))
    plt.close()
