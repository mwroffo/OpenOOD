#!/bin/bash
dataset='osr_cub150/cub150_seed1.yml'
ood_dataset='osr_cub150/cub150_seed1_osr.yml'
network='vit'


# path='results/cub150_seed1_resnet18_224x224_base_e100_lr0.1_pixmix/s0/best_epoch95_acc0.6185.ckpt'
path='results/cub150_seed1_vit-b-16_base_e100_lr0.1_default/s0/best_epoch68_acc0.7664.ckpt'


python3 main.py \
    --config configs/datasets/$dataset \
    configs/datasets/$ood_dataset \
    configs/networks/$network.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/uncle_sam.yml \
    configs/postprocessors/react.yml \
    --network.pretrained True \
    --network.checkpoint $path  