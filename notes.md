## Download data and checkpoints
scripts/download/download.sh

- Had to run download.sh to grab all the results/data
- This will get all the datasets and pretrained model checkpoints we will use later

## Setup for cifar6_ood
Looking at the config in scripts/osr/openmax/sweep_osr.py

- It sets up the Dataset, OOD Dataset, Network, Checkpoint Path
- Cifar6 involves ID (6 classes) and OOD/OSR set (4 classes) from the CIFAR10 dataset

configs/datasets/osr_cifar6/cifar6_seed1.yml

- Sets up ID dataset
- 6 classes (0-5)
- Split into train (30k), val (617), test (5383) splits (~91% of total dataset)

configs/datasets/osr_cifar6/cifar6_seed1_osr.yml
- Sets up OOD/OSR dataset
- has val & osr splits
- validation = same file as ID test from above
- OSR = 3617 (~9% of total dataset)
- 891 cat, 892 dog, 920 bird, 914 ship (none of the classes seen before)

## We basically want to recreate this for OSR with CUBs

## Was trying to run scripts/osr/openmax/sweep_osr.py on cifar6
First had to setup the dataset config for cifar6 in openood/evaluation_api/datasets.py
And set the num_classes dict in info.py

## We were also missing the model checkppoint
So had to run baseline training against the ID set, only 6 classes
./scripts/osr/openmax/train_on_cifar6.sh 
-- training took like 45 minutes, could change the number of epochs down from 100

## Then we should be able to run the OSR test
I forgot to activate my venv so I ran into some package issues, but activating the venv fixed it
./scripts/osr/openmax/my_osr.sh
```
Computing metrics on cifar4 dataset...
FPR@95: 70.52, AUROC: 85.10 AUPR_IN: 86.86, AUPR_OUT: 77.16
ACC: 94.26
──────────────────────────────────────────────────────────────────────
Computing mean metrics...
FPR@95: 70.52, AUROC: 85.10 AUPR_IN: 86.86, AUPR_OUT: 77.16
ACC: 94.26
──────────────────────────────────────────────────────────────────────
Time used for eval_ood: 3s
Completed!
```

# Setup ID/OOD split for CUBs
- Split into ID (150 classes) and OOD (50 classes)
- Then split ID into train/val/test
- This script will write out the image split files
- python3 cubs_osr_split.py

# Training baseline model on ID set
./scripts/osr/openmax/train_on_cubs.sh
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 784.00 MiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use.
```

- Had to drop the batch size from 128 to 16
- Switched to resnet18_224x224 arch and use the pretrained checkpoint

Was able to achieve 61% accuracy on the CUBS ID dataset after 50 epochs

```
Epoch 049 | Time   855s | Train Loss 1.1032 | Val Loss 1.517 | Val Acc 61.15
Epoch 050: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 391/391 [00:15<00:00, 24.47it/s]
Eval: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 54/54 [00:01<00:00, 46.36it/s]

Epoch 050 | Time   873s | Train Loss 0.9701 | Val Loss 1.517 | Val Acc 59.98
Training Completed! Best accuracy: 61.15 at epoch 49
──────────────────────────────────────────────────────────────────────
Start testing...
Eval: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 109/109 [00:02<00:00, 41.93it/s]

Complete Evaluation, Last accuracy 61.55
```

## Running OSR Evaluation
I was then able to run scripts/osr/openmax/cubs150_osr.sh
```
Accuracy 11.44%
──────────────────────────────────────────────────────────────────────
Performing inference on cub150_seed1 dataset...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 109/109 [00:03<00:00, 33.71it/s]
Processing osr...
Performing inference on cub50 dataset...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 24/24 [00:07<00:00,  3.10it/s]
Computing metrics on cub50 dataset...
FPR@95: 93.05, AUROC: 50.37 AUPR_IN: 40.18, AUPR_OUT: 59.61
ACC: 11.44
──────────────────────────────────────────────────────────────────────
Computing mean metrics...
FPR@95: 93.05, AUROC: 50.37 AUPR_IN: 40.18, AUPR_OUT: 59.61
ACC: 11.44
──────────────────────────────────────────────────────────────────────
Time used for eval_ood: 11s
Completed!
````