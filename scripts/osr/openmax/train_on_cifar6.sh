#!/bin/bash

python3 main.py \
    --config configs/datasets/osr_cifar6/cifar6_seed1.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml