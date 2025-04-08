# necessary imports
import torch
# import os, sys
# sys.path.append('./OpenOOD/')
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32  # just a wrapper around the ResNet

# ----------------------
# DEVICE CONFIGURATION
# ----------------------

# ------------------------
# DEVICE SETUP
# ------------------------
# Prefer CUDA, then MPS, then fallback to CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ----------------------
# LOAD TRAINED MODEL
# ----------------------

# Load pretrained ResNet18 model for CIFAR-10
checkpoint_path = 'data/cifar10/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
net = ResNet18_32x32(num_classes=10)
net.load_state_dict(torch.load(checkpoint_path, map_location=device))
net.to(device)
net.eval()  # set model to evaluation mode

# ----------------------
# SELECT POSTPROCESSOR
# ----------------------

# @title choose an implemented postprocessor
postprocessor_name = "react"  # @param ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"] {allow-input: true}
postprocessor_name = "klm"  # @param ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"] {allow-input: true}
postprocessor_name = "odin"  # @param ["openmax", "msp", "temp_scaling", "odin", "mds", "mds_ensemble", "rmds", "gram", "ebo", "gradnorm", "react", "mls", "klm", "vim", "knn", "dice", "rankfeat", "ash", "she"] {allow-input: true}
# ----------------------
# NOTES:
# ----------------------
# 1) The evaluator will automatically download the required datasets given the
#    ID dataset specified by `id_name`
#
# 2) Passing the `postprocessor_name` will use an implemented postprocessor. To
#    use your own postprocessor, just make sure that it inherits the BasePostprocessor
#    class (see openood/postprocessors/base_postprocessor.py) and pass it to the
#    `postprocessor` argument.
#
# 3) `config_root` points to the directory with OpenOOD's configurations for the
#    postprocessors. By default the evaluator will look for the configs that come
#    with the OpenOOD module. If you want to use custom configs, clone the repo locally
#    and make modifications to OpenOOD/configs.
#
# 4) As you will see when executing this cell, during the initialization the evaluator
#    will automatically run hyperparameter search on ID/OOD validation data (if applicable).
#    If you want to use a postprocessor with specific hyperparams, you need
#    to clone the OpenOOD repo (or just download the configs folder in the repo).
#    Then a) specify the hyperparams and b) set APS_mode to False in the respective postprocessor
#    config.

# ----------------------
# INITIALIZE EVALUATOR
# ----------------------

evaluator = Evaluator(
    net,                                 # The trained neural network
    id_name='cifar10',                   # The in-distribution dataset
    data_root='./data',                  # Path to where datasets are/will be downloaded
    config_root=None,                    # Use default configs unless specified
    preprocessor=None,                   # Use default transforms for CIFAR-10
    postprocessor_name=postprocessor_name,  # Choose one of the built-in postprocessors
    postprocessor=None,                  # Pass a custom one if desired (should inherit BasePostprocessor)
    batch_size=200,                      # Batch size for evaluation (tune based on device memory)
    shuffle=False,                       # Evaluation is deterministic
    num_workers=0                        # Set >0 for faster loading on local machines (leave 0 for Colab)
)

# ----------------------
# RUN EVALUATION
# ----------------------

# Perform standard OOD detection (not full-spectrum OOD)
# This returns a metrics dataframe (AUROC, AUPR, FPR@95, etc.)
metrics = evaluator.eval_ood(fsood=False)

# ----------------------
# ACCESSING RESULTS
# ----------------------

# evaluator.metrics: contains evaluation metrics per OOD dataset
# evaluator.scores: contains raw OOD scores and predictions for further analysis

print('Components within evaluator.metrics:\t', evaluator.metrics.keys())
print('Components within evaluator.scores:\t', evaluator.scores.keys())
print('')
print('The predicted ID class of the first 5 samples of CIFAR-100:\t',
      evaluator.scores['ood']['near']['cifar100'][0][:5])
print('The OOD score of the first 5 samples of CIFAR-100:\t',
      evaluator.scores['ood']['near']['cifar100'][1][:5])

# For further inspection, you can plot score histograms or compare confidence scores
# between ID and OOD samples from evaluator.scores

# 5. Extending OpenOOD for your own research/development

# We try to make OpenOOD extensible and convenient for everyone.


# You can evaluate your own trained model as long as it has necessary functions/methods that help it work with the postprocessors (see OpenOOD/openood/resnet18_32x32.py for example).


# You can also design your own postprocessor by inheriting the base class (OpenOOD/openood/postprocessors/base_postprocessor.py), and the resulting method can be readily evaluated with OpenOOD.


# Feel free to reach out to us if you have furthur suggestions on making OpenOOD more general and easy-to-use!