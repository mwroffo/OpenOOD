#!/bin/bash


path='results/checkpoints/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
python3 main.py \
    --config configs/datasets/osr_cifar6/cifar6_seed1.yml \
    configs/networks/vit_train.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint $path