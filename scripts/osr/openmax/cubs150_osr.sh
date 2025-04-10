#!/bin/bash
dataset='osr_cub150/cub150_seed1.yml'
ood_dataset='osr_cub150/cub150_seed1_osr.yml'
network='resnet18_224x224'

# This checkpoint is saved in our shared google drive
path='results/cub150_seed1_resnet18_224x224_base_e50_lr0.1_default/s0/best_epoch49_acc0.6115.ckpt'

python3 main.py \
    --config configs/datasets/$dataset \
    configs/datasets/$ood_dataset \
    configs/networks/$network.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/openmax.yml \
    --network.pretrained True \
    --network.checkpoint $path 