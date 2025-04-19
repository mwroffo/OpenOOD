#!/bin/bash

# Uploaded the imagenet pretrained checkpoint to shared google drive
# Also saved the checkpoint output of this training to google drive
path='results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best_epoch89_acc0.8500.ckpt'
python3 main.py \
    --config configs/datasets/osr_cub150/cub150_seed1.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/randaugment_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint $path