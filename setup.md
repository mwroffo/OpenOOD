For our first milestone, we will focus on 

[] applying OpenOOD methods to the CUB-200-2011 [1] dataset,
focusing on the [] post hoc and [] data augmentation meth-
ods, as these performed best. Nature photography
datasets such as CUB-200-2011 often exhibit back-
ground noise, where training samples contain extrane-
ous objects alongside the target class, and bounding-
box annotations contain pixels that are not part of the
object. Therefore, a primary challenge for our project
will be [] minimizing performance degradation caused by
the model learning spurious background features, rather
than features of the annotated object. One way to mit-
igate the impact of background noise would be to [] cap-
ture background representations and take them into ac-
count when computing class scores, perhaps via cluster-
ing methods.

### Access the Google Compute Engine VM via SSH (No GPU)
- At https://console.cloud.google.com/compute/metadata , do "Edit" then add your SSH key. Then "Save"
- Under "VM instances", see the "External IP address"
- then you can login to it using the ssh client of your choice. `ssh -i PATH_TO_PRIVATE_KEY USERNAME@EXTERNAL_IP`

I created the directory /682-proj with open permissions so we can access the data without duplicating it. Also we can each create a personal directory here:

```
(venv) _mexus@instance-20250406-002240:/682-proj/michael/libMR$ ls -la /682-proj/
total 16
drwxrwxrwx  4 root    root    4096 Apr  6 01:00 .
drwxr-xr-x 21 root    root    4096 Apr  6 00:47 ..
drwxrwxrwx  4 mwroffo mwroffo 4096 Apr  6 02:27 michael
drwxrwxrwx  4 root    root    4096 Apr  6 01:40 shared
```

The ubuntu VM has python 3.10. It does not have GPU but could be allocated on a new VM if necessary.

Install steps as follows

```
python3 -m pip install -e .  # install openood
python3 -m pip install cython numpy libmr

## if pip install fails, try cloning libmr and doing 

pip install .

## But you shouldn't need that.

## Support ipynb
python3 -m pip install ipykernel
```

Remember to run ssh-keygen on the VM and copy the public key to your github.

Just for the record, I did have to do some sysadminish steps to include necessary build tools:
```
## build tools for installing libmr
sudo apt install -y build-essential python3.10-dev
sudo apt install -y libgl1 unzip
```

### Verify python version
```
python3.10 --version
Python 3.10.12
```

### New venv
```
python3.10 -m venv venv
source venv/bin/activate
```

### Run install script
#### Specify compatible packages
In setup.py, add `faiss-cpu>=1.7.2`, `numpy==1.25.2`, `statsmodels`
```
install_requires=[
    ...
    'faiss-cpu>=1.7.2',
    'numpy==1.25.2',
    'statsmodels'
    ...
]
```
#### Install
From OpenOOD repo root
```
python3.10 -m pip install -e .
python3.10 -m pip install libmr
```

### Download datasets
I added the downloads to the .gitignore as well
```
gdown 1byGeYxM_PlLjT72wZsMQvP6popJeWBgt
unzip cifar10_res18_v1.5.zip
```

### Run tutorial.py
```
python3.10 tutorial.py
```

### Tutorial Results
```
Computing metrics on cifar100 dataset...
FPR@95: 75.50, AUROC: 85.24 AUPR_IN: 80.67, AUPR_OUT: 85.83
──────────────────────────────────────────────────────────────────────

Performing inference on tin dataset...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 39/39 [00:02<00:00, 14.35it/s]
Computing metrics on tin dataset...
FPR@95: 67.62, AUROC: 87.70 AUPR_IN: 85.16, AUPR_OUT: 86.93
──────────────────────────────────────────────────────────────────────

Computing mean metrics...
FPR@95: 71.56, AUROC: 86.47 AUPR_IN: 82.91, AUPR_OUT: 86.38
──────────────────────────────────────────────────────────────────────

Processing far ood...
Performing inference on mnist dataset...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 350/350 [00:21<00:00, 16.26it/s]
Computing metrics on mnist dataset...
FPR@95: 18.42, AUROC: 95.38 AUPR_IN: 75.86, AUPR_OUT: 99.31
──────────────────────────────────────────────────────────────────────

Performing inference on svhn dataset...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 131/131 [00:07<00:00, 16.94it/s]
Computing metrics on svhn dataset...
FPR@95: 44.14, AUROC: 90.01 AUPR_IN: 75.57, AUPR_OUT: 95.45
──────────────────────────────────────────────────────────────────────

Performing inference on texture dataset...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 29/29 [00:09<00:00,  2.92it/s]
Computing metrics on texture dataset...
FPR@95: 67.36, AUROC: 87.27 AUPR_IN: 88.35, AUPR_OUT: 82.30
──────────────────────────────────────────────────────────────────────

Performing inference on places365 dataset...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 176/176 [00:19<00:00,  8.85it/s]
Computing metrics on places365 dataset...
FPR@95: 39.82, AUROC: 91.40 AUPR_IN: 71.91, AUPR_OUT: 97.39
──────────────────────────────────────────────────────────────────────

Computing mean metrics...
FPR@95: 42.44, AUROC: 91.02 AUPR_IN: 77.92, AUPR_OUT: 93.61
──────────────────────────────────────────────────────────────────────

ID Acc Eval: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 45/45 [00:03<00:00, 14.21it/s]
           FPR@95  AUROC  AUPR_IN  AUPR_OUT   ACC
cifar100    75.50  85.24    80.67     85.83 95.22
tin         67.62  87.70    85.16     86.93 95.22
nearood     71.56  86.47    82.91     86.38 95.22
mnist       18.42  95.38    75.86     99.31 95.22
svhn        44.14  90.01    75.57     95.45 95.22
texture     67.36  87.27    88.35     82.30 95.22
places365   39.82  91.40    71.91     97.39 95.22
farood      42.44  91.02    77.92     93.61 95.22
Components within evaluator.metrics:     dict_keys(['id_acc', 'csid_acc', 'ood', 'fsood'])
Components within evaluator.scores:      dict_keys(['id', 'csid', 'ood', 'id_preds', 'id_labels', 'csid_preds', 'csid_labels'])

The predicted ID class of the first 5 samples of CIFAR-100:      [9 9 9 9 9]
The OOD score of the first 5 samples of CIFAR-100:       [5.151 5.213 6.402 6.655 5.155]
```