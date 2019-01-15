import datetime
import itertools
import os

from pyapt_scripts import pyapt, launchutils
from pyapt_scripts import pytorchapt
import numpy as np

train_dataset = 'primitives'
train_split = 'train'

hidden_layer_nbs = [2, 8]
hidden_neuron_nbs = [256]

epochs = 500

batch_size = 4
optimizers = ['adam']
lrs = [0.001]
momentum = 0.9
parallel_args = []
pytorch_version = 4
mini_factors = [0.1]  # TODO mini_factor = None
sdf_point_nb = 800

for (lr, optimizer, mini_factor, hidden_layer_nb, hidden_neuron_nb) in itertools.product(
        lrs, optimizers,  mini_factors, hidden_layer_nbs, hidden_neuron_nbs):
        args = {
            'batch_size': batch_size,
            'display_freq': 100,
            'epoch_display_freq': 20,
            'hidden_layer_nb': hidden_layer_nb,
            'hidden_neuron_nb': hidden_neuron_nb,
            'lr': lr,
            'mini_factor': mini_factor,
            'momentum': momentum,
            'optimizer': optimizer,
            'dataset': train_dataset,
            'snapshot': 100,
            'sdf_point_nb': sdf_point_nb,
            'train_split': train_split,
        }
        arguments = launchutils.clean_args(args)
        parallel_args.append(arguments)

shell_vars = pytorchapt.get_pytorch_shell_vars(pytorch_version)

prepend_cmd = []
cd_project_folder = 'cd /sequoia/data2/yhasson/code/shapesfd'
conda_activate = 'source activate pytorch-env'
prepend_cmd.extend([cd_project_folder, conda_activate])

python_cmd = 'preprocess.py'

queues = ['titan.q', 'chronos.q', 'zeus.q']
pyapt.apt_run(
    python_cmd,
    parallel_args=parallel_args,
    queues=queues,
    shell_var=shell_vars,
    prepend_cmd=prepend_cmd,
    group_by=1,
    memory=12000,
    memory_hard=2000000,
    multi_threading=1)
