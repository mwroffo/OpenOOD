
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
net.eval()
ood_splits = ['nearood', 'farood']
for ood_split in ood_splits:
    for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
        dataiter = iter(ood_dl)

        logits_list = []
        feature_list = []
        label_list = []

        for i in tqdm(range(1,
                        len(dataiter) + 1),
                        desc='Extracting results...',
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

        save_arr_to_dir(logits_arr, osp.join(save_root, ood_split, f'{dataset_name}_logits.npy'))
        save_arr_to_dir(feature_arr, osp.join(save_root, ood_split, f'{dataset_name}_feature.npy'))
        save_arr_to_dir(label_arr, osp.join(save_root, ood_split, f'{dataset_name}_labels.npy'))


# build msp method (pass in pre-saved logits)
def msp_postprocess(logits):
    score = torch.softmax(logits, dim=1)
    conf, pred = torch.max(score, dim=1)
    return pred, conf

# load logits, feature, label for this benchmark
results = dict()
# for id
modes = ['val', 'test']
results['id'] = dict()
for mode in modes:
    results['id'][mode] = dict()
    results['id'][mode]['feature'] = np.load(osp.join(save_root, 'id', f'{mode}_feature.npy'))
    results['id'][mode]['logits'] = np.load(osp.join(save_root, 'id', f'{mode}_logits.npy'))
    results['id'][mode]['labels'] = np.load(osp.join(save_root, 'id', f'{mode}_labels.npy'))

# for ood
split_types = ['nearood', 'farood']
for split_type in split_types:
    results[split_type] = dict()
    dataset_names = config['ood_dataset'][split_type].datasets
    for dataset_name in dataset_names:
        results[split_type][dataset_name] = dict()
        results[split_type][dataset_name]['feature'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_feature.npy'))
        results[split_type][dataset_name]['logits'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_logits.npy'))
        results[split_type][dataset_name]['labels'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_labels.npy'))

def print_nested_dict(dict_obj, indent = 0):
    ''' Pretty Print nested dictionary with given indent level
    '''
    # Iterate over all key-value pairs of dictionary
    for key, value in dict_obj.items():
        # If value is dict type, then print nested dict
        if isinstance(value, dict):
            print(' ' * indent, key, ':', '{')
            print_nested_dict(value, indent + 2)
            print(' ' * indent, '}')
        else:
            print(' ' * indent, key, ':', value.shape)
print_nested_dict(results)


# get pred, conf, gt from MSP postprocessor (can change to your custom_postprocessor here)
postprocess_results = dict()
# id
modes = ['val', 'test']
postprocess_results['id'] = dict()
for mode in modes:
    pred, conf = msp_postprocess(torch.from_numpy(results['id'][mode]['logits']))
    pred, conf = pred.numpy(), conf.numpy()
    gt = results['id'][mode]['labels']
    postprocess_results['id'][mode] = [pred, conf, gt]

# ood
split_types = ['nearood', 'farood']
for split_type in split_types:
    postprocess_results[split_type] = dict()
    dataset_names = config['ood_dataset'][split_type].datasets
    for dataset_name in dataset_names:
        pred, conf = msp_postprocess(torch.from_numpy(results[split_type][dataset_name]['logits']))
        pred, conf = pred.numpy(), conf.numpy()
        gt = results[split_type][dataset_name]['labels']
        gt = -1 * np.ones_like(gt)   # hard set to -1 here
        postprocess_results[split_type][dataset_name] = [pred, conf, gt]
def print_all_metrics(metrics):
    [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
    print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
            end=' ',
            flush=True)
    print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
        100 * aupr_in, 100 * aupr_out),
            flush=True)
    print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
    print(u'\u2500' * 70, flush=True)

from openood.evaluators.metrics import compute_all_metrics
def eval_ood(postprocess_results):
    [id_pred, id_conf, id_gt] = postprocess_results['id']['test']
    split_types = ['nearood', 'farood']

    for split_type in split_types:
        metrics_list = []
        print(f"Performing evaluation on {split_type} datasets...")
        dataset_names = config['ood_dataset'][split_type].datasets

        for dataset_name in dataset_names:
            [ood_pred, ood_conf, ood_gt] = postprocess_results[split_type][dataset_name]

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            print_all_metrics(ood_metrics)
            metrics_list.append(ood_metrics)
        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        print_all_metrics(metrics_mean)

eval_ood(postprocess_results)