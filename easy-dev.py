# === Imports ===
import os
import os.path as osp
import torch
import numpy as np
from tqdm import tqdm

# OpenOOD-specific modules
from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.evaluators.metrics import compute_all_metrics

# === Load Config Files ===
# Combine multiple YAML config files into one master config object
config_files = [
    './configs/datasets/cifar10/cifar10.yml',
    './configs/datasets/cifar10/cifar10_ood.yml',
    './configs/networks/resnet18_32x32.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/msp.yml',
]
config = config.Config(*config_files)

# === Modify Config Dynamically ===
config.network.checkpoint = './cifar10_resnet18_32x32_base_e100_lr0.1_default/s2/best.ckpt'
config.network.pretrained = True
config.num_workers = 0
config.save_output = False
config.parse_refs()

# === DEVICE CONFIGURATION ===
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# === LOAD DATALOADERS ===
id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)

# === INITIALIZE NETWORK ===
net = get_network(config.network).to(device)

# === INITIALIZE EVALUATOR ===
evaluator = get_evaluator(config)

# === Helper Function: Save numpy arrays to disk ===
def save_arr_to_dir(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        np.save(f, arr)

# === Extract Features and Logits from the Network and Save ===
save_root = f'./results/{config.exp_name}'
net.eval()

for mode in ['test', 'val']:
    dl = id_loader_dict[mode]
    dataiter = iter(dl)
    logits_list, feature_list, label_list = [], [], []

    for _ in tqdm(range(len(dataiter)), desc='Extracting results...', position=0, leave=True):
        batch = next(dataiter)
        data = batch['data'].to(device)
        label = batch['label']
        with torch.no_grad():
            logits_cls, feature = net(data, return_feature=True)
        logits_list.append(logits_cls.cpu().numpy())
        feature_list.append(feature.cpu().numpy())
        label_list.append(label.numpy())

    save_arr_to_dir(np.concatenate(logits_list), osp.join(save_root, 'id', f'{mode}_logits.npy'))
    save_arr_to_dir(np.concatenate(feature_list), osp.join(save_root, 'id', f'{mode}_feature.npy'))
    save_arr_to_dir(np.concatenate(label_list), osp.join(save_root, 'id', f'{mode}_labels.npy'))

for ood_split in ['nearood', 'farood']:
    for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
        dataiter = iter(ood_dl)
        logits_list, feature_list, label_list = [], [], []

        for _ in tqdm(range(len(dataiter)), desc='Extracting results...', position=0, leave=True):
            batch = next(dataiter)
            data = batch['data'].to(device)
            label = batch['label']
            with torch.no_grad():
                logits_cls, feature = net(data, return_feature=True)
            logits_list.append(logits_cls.cpu().numpy())
            feature_list.append(feature.cpu().numpy())
            label_list.append(label.numpy())

        save_arr_to_dir(np.concatenate(logits_list), osp.join(save_root, ood_split, f'{dataset_name}_logits.npy'))
        save_arr_to_dir(np.concatenate(feature_list), osp.join(save_root, ood_split, f'{dataset_name}_feature.npy'))
        save_arr_to_dir(np.concatenate(label_list), osp.join(save_root, ood_split, f'{dataset_name}_labels.npy'))

# === MSP Postprocessing ===
def msp_postprocess(logits):
    score = torch.softmax(logits, dim=1)
    conf, pred = torch.max(score, dim=1)
    return pred, conf

# === Load Pre-Saved Results ===
results = dict()

results['id'] = {}
for mode in ['val', 'test']:
    results['id'][mode] = {
        'feature': np.load(osp.join(save_root, 'id', f'{mode}_feature.npy')),
        'logits': np.load(osp.join(save_root, 'id', f'{mode}_logits.npy')),
        'labels': np.load(osp.join(save_root, 'id', f'{mode}_labels.npy')),
    }

for split_type in ['nearood', 'farood']:
    results[split_type] = {}
    dataset_names = config['ood_dataset'][split_type].datasets
    for dataset_name in dataset_names:
        results[split_type][dataset_name] = {
            'feature': np.load(osp.join(save_root, split_type, f'{dataset_name}_feature.npy')),
            'logits': np.load(osp.join(save_root, split_type, f'{dataset_name}_logits.npy')),
            'labels': np.load(osp.join(save_root, split_type, f'{dataset_name}_labels.npy')),
        }

# === Helper: Print Nested Dictionary Shape Summary ===
def print_nested_dict(dict_obj, indent=0):
    for key, value in dict_obj.items():
        if isinstance(value, dict):
            print(' ' * indent, key, ':', '{')
            print_nested_dict(value, indent + 2)
            print(' ' * indent, '}')
        else:
            print(' ' * indent, key, ':', value.shape)

print_nested_dict(results)

# === Apply Postprocessing to Loaded Results ===
postprocess_results = {'id': {}}

for mode in ['val', 'test']:
    pred, conf = msp_postprocess(torch.from_numpy(results['id'][mode]['logits']))
    postprocess_results['id'][mode] = [
        pred.numpy(),
        conf.numpy(),
        results['id'][mode]['labels']
    ]

for split_type in ['nearood', 'farood']:
    postprocess_results[split_type] = {}
    dataset_names = config['ood_dataset'][split_type].datasets
    for dataset_name in dataset_names:
        pred, conf = msp_postprocess(torch.from_numpy(results[split_type][dataset_name]['logits']))
        gt = -1 * np.ones_like(results[split_type][dataset_name]['labels'])
        postprocess_results[split_type][dataset_name] = [pred.numpy(), conf.numpy(), gt]

# === Print Evaluation Metrics ===
def print_all_metrics(metrics):
    [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
    print(f'FPR@95: {100 * fpr:.2f}, AUROC: {100 * auroc:.2f}', end=' ', flush=True)
    print(f'AUPR_IN: {100 * aupr_in:.2f}, AUPR_OUT: {100 * aupr_out:.2f}', flush=True)
    print(f'ACC: {100 * accuracy:.2f}', flush=True)
    # print(f'CCR: {100 * accuracy:.2f}')
    print(u'─' * 70, flush=True)

# === Evaluate OOD Detection Performance ===
def eval_ood(postprocess_results):
    id_pred, id_conf, id_gt = postprocess_results['id']['test']
    for split_type in ['nearood', 'farood']:
        print(f"Performing evaluation on {split_type} datasets...")
        metrics_list = []
        for dataset_name in config['ood_dataset'][split_type].datasets:
            ood_pred, ood_conf, ood_gt = postprocess_results[split_type][dataset_name]
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

# Run the evaluation
eval_ood(postprocess_results)
