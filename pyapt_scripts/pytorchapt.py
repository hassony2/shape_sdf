def get_pytorch_shell_vars(version=3):
    shell_vars = []

    # Add local python packages
    python_paths = [
        '/sequoia/data1/yhasson/code/hand-cnns',
        '/sequoia/data1/yhasson/code/myana',
        '/sequoia/data1/yhasson/code/pose_3d/handobjectdatasets',
        '/sequoia/data1/yhasson/code/pose_3d/mano_render',
        '/sequoia/data1/yhasson/code/actiondatasets',
        '/sequoia/data1/yhasson/code/videotransforms_pytorch',
        '/sequoia/data1/yhasson/code/custom_formats',
        '/sequoia/data1/yhasson/code/pose_cluster/full_extraction',
        '/sequoia/data1/yhasson/code/third_party'
    ]
    python_path_string = ':'.join(python_paths)

    shell_vars.append(('PYTHONPATH', python_path_string))

    # Add conda env variable
    conda_var = ('CONDA_PREFIX',
                 '/sequoia/data3/yhasson/miniconda3/envs/pytorch-env')
    shell_vars.append(conda_var)

    # Add PATH env variable
    path_paths = [
        '/sequoia/data1/yhasson/perso_lib/bin',
        '/sequoia/data3/yhasson/miniconda3/bin'
    ]
    shell_vars.append(('PATH', ':'.join(path_paths)))
    return shell_vars
