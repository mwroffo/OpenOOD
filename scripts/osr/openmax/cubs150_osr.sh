#!/bin/bash
dataset='osr_cub150/cub150_seed1.yml'
ood_dataset='osr_cub150/cub150_seed1_osr.yml'
# network='resnet18_224x224'
network='vit'

# These checkpoints are saved in our shared google drive
# Trained for 100 epochs 
path='results/cub150_seed1_resnet18_224x224_base_e100_lr0.1_pixmix/s0/best_epoch95_acc0.6185.ckpt'
# 50 epochs
# path='results/cub150_seed1_resnet18_224x224_base_e50_lr0.1_default/s0/best_epoch49_acc0.6115.ckpt'

python3 main.py \
    --config configs/datasets/$dataset \
    configs/datasets/$ood_dataset \
    configs/networks/vit.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vit_mahalanobis.yml \
    
    # --network.pretrained True \
    # --network.checkpoint $path  \
    # --mark pixmix