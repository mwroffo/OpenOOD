#!/bin/bash

# This file will fine tune the pretrained ResNet checkpoint on the CUBs dataset,
# that was preproccessed by cropping the subject using the bounding box coordinates, and
# then using SAM to blur the background.

# You will also need to create the appropriate benchmark_imglist/ files that the dataset configs reference.
# The splits are the exact same as the original cub150/50 split, just with the image directories changed.
# i.e. copy+paste, find/replace all: CUB_200_2011 -> CUB_200_2011_crop_and_blur
# data/CUB_200_2011/images/002.Laysan_Albatross/Laysan_Albatross_0002_1027.jpg -1
# data/CUB_200_2011_crop_and_blur/images/002.Laysan_Albatross/Laysan_Albatross_0002_1027.jpg -1


# Pretrained imagenet checkpoint
path='results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best_epoch89_acc0.8500.ckpt'
python3 main.py \
    --config configs/datasets/osr_cub150/cub150_seed1_crop_and_blur.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint $path