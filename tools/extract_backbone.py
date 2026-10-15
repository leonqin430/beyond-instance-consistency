"""Extract the pre-trained backbone of a checkpoint of main_moco.py / main_dino.py.

The output is a plain state dict:
  * MoCo v2: torchvision ResNet-50 keys (conv1.*, bn1.*, layer1.*, ...). It can be loaded with
    `torchvision.models.resnet50().load_state_dict(sd, strict=False)` (no fc) or as the pre-trained
    backbone of MMDetection / MMRotate, e.g.
    `backbone=dict(init_cfg=dict(type='Pretrained', checkpoint='resnet50_moco.pth'))`.
  * DINO: keys of `models_vit.vit_small` (teacher by default).

Usage:
    python tools/extract_backbone.py output_dir/checkpoint.pth resnet50_moco.pth
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.misc import load_backbone_state_dict  # noqa: E402


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract the backbone of a pre-training checkpoint')
    parser.add_argument('input', type=str, help='checkpoint of main_moco.py or main_dino.py')
    parser.add_argument('output', type=str, help='output state dict')
    parser.add_argument('--checkpoint_key', default=None, type=str,
                        help='default: teacher for DINO, state_dict for MoCo')
    args = parser.parse_args()

    state_dict = load_backbone_state_dict(args.input, args.checkpoint_key)
    torch.save(state_dict, args.output)
    print(f'Saved {len(state_dict)} tensors to {args.output}')
