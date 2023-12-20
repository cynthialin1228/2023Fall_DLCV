#!/bin/bash
python3 ./p3/inference.py --input_dir $1 --output_dir $2 --svhn_model "./svhn.ckpt" --usps_model "./usps.ckpt"
# TODO - run your inference Python3 code