#!/bin/bash
dataset='osr_cub150/cub150_seed1.yml'
# dataset='osr_cifar6/cifar6_seed1.yml'
ood_dataset='osr_cub150/cub150_seed1_osr.yml'
# ood_dataset='osr_cifar6/cifar6_seed1_osr.yml'
network='dino_vits'

python3 main.py \
    --config configs/datasets/$dataset \
    configs/datasets/$ood_dataset \
    configs/networks/$network.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/gram.yml