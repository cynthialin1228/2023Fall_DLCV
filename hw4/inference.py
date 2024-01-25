import torch
from nerf_pl.utils import *
from collections import defaultdict
import os
from nerf_pl.models.rendering import *
from nerf_pl.models.nerf import *
from dataset import KlevrDataset
import argparse


def get_args():
    parser = argparse.ArgumentParser(
        description="nerf_pl"
    )
    parser.add_argument(
        "--input_dir", "-i", type=str, default="./dataset", help="path to input dir"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default="./output", help="output image directory"
    )
    parser.add_argument(
        "--model_path", "-m", type=str, default="./nerf_pl_model.ckpt", help="model of nerf"
    )
    return parser.parse_args()


@torch.no_grad()
def f(rays, models, embeddings, N_samples, N_importance, use_disp, chunk=1024*32*4):
    B = rays.shape[0]
    results = defaultdict(list)
    for i in range(0, B, chunk):
        rendered_ray_chunks = render_rays(models, embeddings, rays[i:i+chunk], N_samples,
                        use_disp, 0, 0, N_importance, chunk, True, test_time=True)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]
    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def infer(input_dir, output_dir, model_path):
    torch.backends.cudnn.benchmark = True

    os.makedirs(output_dir, exist_ok=True)
    dataset = KlevrDataset(root_dir=input_dir, split='test')

    embedding_xyz = Embedding(3, 10)
    embedding_dir = Embedding(3, 4)
    nerf_coarse = NeRF()
    nerf_fine = NeRF()

    ckpt_path = model_path
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
    img_wh = (256, 256)

    for idx, image_id in enumerate(dataset.split_ids):
        input_filename = f'{image_id:05d}'

        sample = dataset[idx]
        rays = sample['rays'].cuda()
        results = f(rays, models, embeddings, N_samples, N_importance, use_disp, chunk)

        img_pred = results['rgb_fine'].view(img_wh[1], img_wh[0], 3).cpu().numpy()
        img_to_save = Image.fromarray((img_pred * 255).astype('uint8'))
        img_to_save.save(os.path.join(output_dir, f'{input_filename}.png'))

    return


if __name__ == "__main__":
    args = get_args()
    infer(args.input_dir, args.output_dir, args.model_path)