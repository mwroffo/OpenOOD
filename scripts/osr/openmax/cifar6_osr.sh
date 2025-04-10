#!/bin/bash
dataset='osr_cifar6/cifar6_seed1.yml'
ood_dataset='osr_cifar6/cifar6_seed1_osr.yml'
network='resnet18_32x32'

path='results/cifar6_seed1_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch99_acc0.9741.ckpt'

python3 main.py \
    --config configs/datasets/$dataset \
    configs/datasets/$ood_dataset \
    configs/networks/$network.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/openmax.yml \
    --network.pretrained True \
    --network.checkpoint $path 