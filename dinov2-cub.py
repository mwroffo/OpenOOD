#!/usr/bin/env python3
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms as trn

from openood.evaluation_api import Evaluator

# -----------------------------------------------------------------------------
# 1) Wrap DINOv2 backbone with a simple linear head for C classes
# -----------------------------------------------------------------------------
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # DINOv2 ViT-B/14 has .embed_dim = 768
        self.fc = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, x, return_feature: bool = False):
        features = self.backbone(x)        # [B, D]
        logits = self.fc(features)         # [B, C]
        if return_feature:
            return logits, features
        return logits

# -----------------------------------------------------------------------------
# 2) Argument parsing
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="OOD eval on CUB-150 / CUB-50 with DINOv2")
    p.add_argument('--arch',
                   choices=['vits14','vitb14','vitl14','vitg14'],
                   default='vitb14',
                   help="Which DINOv2 model to load")
    p.add_argument('--postprocessor',
                   choices=['msp','odin','maha','openmax','rmds','vim','ash'],
                   default='msp')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--data-root',   type=str, default='./data',
                   help="Root folder for CUB data")
    p.add_argument('--config-root', type=str, default='./configs',
                   help="Root folder for OpenOOD YAML configs")
    return p.parse_args()

# -----------------------------------------------------------------------------
# 3) Main evaluation logic
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # 3.1 Pull DINOv2 backbone
    model_name = f"dinov2_{args.arch}"
    backbone = torch.hub.load('facebookresearch/dinov2', model_name)

    # 3.2 Build classifier for 150 ID classes
    net = DINOv2Classifier(backbone, num_classes=150)

    # 3.3 Preprocessing pipeline
    preprocessor = trn.Compose([
        trn.Resize(256, interpolation=trn.InterpolationMode.BICUBIC),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize(mean=[0.485,0.456,0.406],
                      std =[0.229,0.224,0.225]),
    ])

    # 3.4 Device setup (CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    net.to(device)
    net.eval()
    cudnn.benchmark = True

    # 3.5 Initialize OpenOOD Evaluator
    evaluator = Evaluator(
        net,
        id_name='osr_cub',                         # name must match configs/datasets/osr_cub150/cub150_seed1.yml
        data_root=args.data_root,                       # where the data lives
        config_root=args.config_root,                   # where your .ymls live
        preprocessor=preprocessor,
        postprocessor_name=args.postprocessor,
        postprocessor=None,                             # pass None to load built‐in
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # 3.6 Run OOD evaluation (this will use the 'osr' split from your ood_dataset YAML)
    metrics = evaluator.eval_ood(fsood=False)

    # 3.7 Print / save results
    print(metrics.to_markdown())

if __name__ == '__main__':
    main()
