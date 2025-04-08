
from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network

from tqdm import tqdm
import numpy as np
import torch
import os.path as osp

import torch

# load config files for cifar10 baseline
config_files = [
    './configs/datasets/cifar10/cifar10.yml',
    './configs/datasets/cifar10/cifar10_ood.yml',
    './configs/networks/resnet18_32x32.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/msp.yml',
]
config = config.Config(*config_files)
# modify config
config.network.checkpoint = './cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt'

config.network.pretrained = True
config.num_workers = 0
config.save_output = False
config.parse_refs()

# -------------------------
# DEVICE CONFIGURATION
# -------------------------
# Prefer CUDA if available, otherwise fallback to MPS or CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# -------------------------
# LOAD DATALOADERS
# -------------------------
# Get in-distribution and out-of-distribution data loaders
id_loader_dict = get_dataloader(config)        # typically includes train/val/test splits
ood_loader_dict = get_ood_dataloader(config)   # typically includes near-OOD and/or far-OOD sets

# -------------------------
# INITIALIZE NETWORK
# -------------------------
# Load the model architecture defined in the config and move to device
net = get_network(config.network)

# -------------------------
# INITIALIZE EVALUATOR
# -------------------------
# Create evaluator instance using provided configuration
# Handles postprocessing, scoring, and metrics computation
evaluator = get_evaluator(config)

import os

def save_arr_to_dir(arr, path):
    # Ensure the parent directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the NumPy array
    with open(path, 'wb') as f:
        np.save(f, arr)
        
save_root = f'./results/{config.exp_name}'

# save id (test & val) results
net.eval()
modes = ['test', 'val']
for mode in modes:
    dl = id_loader_dict[mode]
    dataiter = iter(dl)

    logits_list = []
    feature_list = []
    label_list = []

    for i in tqdm(range(1,
                    len(dataiter) + 1),
                    desc='Extracting reults...',
                    position=0,
                    leave=True):
        batch = next(dataiter)
        data = batch['data'].to(device)
        label = batch['label']
        with torch.no_grad():
            logits_cls, feature = net(data, return_feature=True)
        logits_list.append(logits_cls.data.to('cpu').numpy())
        feature_list.append(feature.data.to('cpu').numpy())
        label_list.append(label.numpy())

    logits_arr = np.concatenate(logits_list)
    feature_arr = np.concatenate(feature_list)
    label_arr = np.concatenate(label_list)
    
    
    save_arr_to_dir(logits_arr, osp.join(save_root, 'id', f'{mode}_logits.npy'))
    save_arr_to_dir(feature_arr, osp.join(save_root, 'id', f'{mode}_feature.npy'))
    save_arr_to_dir(label_arr, osp.join(save_root, 'id', f'{mode}_labels.npy'))


# save ood results
# net.eval()
# ood_splits = ['nearood', 'farood']
# for ood_split in ood_splits:
#     for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
#         dataiter = iter(ood_dl)

#         logits_list = []
#         feature_list = []
#         label_list = []

#         for i in tqdm(range(1,
#                         len(dataiter) + 1),
#                         desc='Extracting reults...',
#                         position=0,
#                         leave=True):
#             batch = next(dataiter)
#             data = batch['data'].to(device)
#             label = batch['label']

#             with torch.no_grad():
#                 logits_cls, feature = net(data, return_feature=True)
#             logits_list.append(logits_cls.data.to('cpu').numpy())
#             feature_list.append(feature.data.to('cpu').numpy())
#             label_list.append(label.numpy())

#         logits_arr = np.concatenate(logits_list)
#         feature_arr = np.concatenate(feature_list)
#         label_arr = np.concatenate(label_list)

#         save_arr_to_dir(logits_arr, osp.join(save_root, ood_split, f'{dataset_name}_logits.npy'))
#         save_arr_to_dir(feature_arr, osp.join(save_root, ood_split, f'{dataset_name}_feature.npy'))
#         save_arr_to_dir(label_arr, osp.join(save_root, ood_split, f'{dataset_name}_labels.npy'))
