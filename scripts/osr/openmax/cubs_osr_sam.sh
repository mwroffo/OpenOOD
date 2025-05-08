#!/bin/bash

# The _sam.yml datasets identify the image sets that were preprocessed 
# w SAM to segment the subject and create a fully black background
# This dataset does not perform very well as the segmentation was somewhat faulty.
# dataset='osr_cub150/cub150_seed1_sam.yml'

# The _blur.yml datasets identify the image sets there were preprocessed 
# w SAM to segment the subject and blur the backgroud
# dataset='osr_cub150/cub150_seed1_blur.yml'

# The _crop_and_blur.yml datasets identify the image sets there were preprocessed 
# by cropping the image to the bounding boxes and using SAM to segment the subject and blur the backgroud
dataset='osr_cub150/cub150_seed1_crop_and_blur.yml'

# Same processing here for the OSR sets
# ood_dataset='osr_cub150/cub150_seed1_osr_sam.yml'
# ood_dataset='osr_cub150/cub150_seed1_osr_blur.yml'
ood_dataset='osr_cub150/cub150_seed1_osr_crop_and_blur.yml'

network='resnet18_224x224'

# Black background
# path='results/cub150_seed1_sam_resnet18_224x224_base_e100_lr0.1_default/s0/best_epoch83_acc0.3357.ckpt'

# Blurred background
# path='results/cub150_seed1_blur_resnet18_224x224_base_e100_lr0.1_default/s0/best_epoch93_acc0.5141.ckpt'

# Cropped image and blurred background
path='results/cub150_seed1_crop_and_blur_resnet18_224x224_base_e100_lr0.1_default/s0/best_epoch94_acc0.3920.ckpt'

python3 main.py \
    --config configs/datasets/$dataset \
    configs/datasets/$ood_dataset \
    configs/networks/$network.yml \
    configs/pipelines/test/test_osr.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/vim.yml \
    --network.pretrained True \
    --network.checkpoint $path