#!/bin/bash
python3 inference.py --input_dir $1 --output_dir $2 --model_path "./nerf_pl_model.ckpt"
# TODO - run your inference Python3 code