#!/bin/bash

# This file will fine tune the pretrained ResNet checkpoint on the CUBs dataset,
# that was preproccessed by cropping the subject using the bounding box coordinates.
# THIS FILE HAS NOT BEEN RUN YET.

# To create the cropped images, run bbox_crop.py


# Uploaded the imagenet pretrained checkpoint to shared google drive
# Also saved the checkpoint output of this training to google drive
# path='results/cub150_seed1_resnet18_224x224_base_e100_lr0.1_default/s0/best_epoch97_acc0.6444.ckpt'
# Trying with base imagenet checkpoint for a little
path='results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best_epoch89_acc0.8500.ckpt'
python3 main.py \
    --config configs/datasets/osr_cub150/cub150_seed1_crop.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/train/baseline.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.pretrained True \
    --network.checkpoint $path