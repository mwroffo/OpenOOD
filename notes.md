# Had to run download/download.sh to grab all the results/data 
idk if this is needed but it helps to see all the references

# Setup for cifar6_ood
scripts/osr/openmax/sweep_osr.py
Dataset, OOD Dataset, Network, Path

configs/datasets/osr_cifar6/cifar6_seed1.yml
Sets up ID dataset
6 classes (0-5)
Split into train (30k), val (617), test (5383) splits (~91%)

configs/datasets/osr_cifar6/cifar6_seed1_osr.yml
Sets up OOD/OSR dataset
has val & osr splits
validation = same file as ID test from above
OSR = 3617 (~9%)
891 cat, 892 dog, 920 bird, 914 ship (none of the classes seen before)

# We basically want to recreate this for OSR with CUBs

# Was trying to run scripts/osr/openmax/sweep_osr.py on cifar6
First had to setup the dataset config for cifar6
And set the num_classes dict

# We were also missing the model checkppoint
So had to run baseline training against the ID set, only 6 classes
./scripts/osr/openmax/train_on_cifar6.sh 
-- training took like 45 minutes, could change the number of epochs down from 100

# Then we should be able to run the OSR test
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
Split into ID (150 classes) and OOD (50 classes)
Then split ID into train/val/test
This script will write out the image split files
python3 cubs_osr_split.py

# Training baseline model on ID set
./scripts/osr/openmax/train_on_cubs.sh
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 784.00 MiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use.
```