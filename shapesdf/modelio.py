from copy import deepcopy
import os
import pickle
import shutil
import traceback
import warnings

import torch

from shapesdf.sdfnet import SFDNet


def reload(checkpoint):
    opt_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint)), 'opt.pkl')
    with open(opt_path, 'rb') as pk_f:
        opts = pickle.load(pk_f)
    model = SFDNet(inter_neurons=[opts['hidden_neuron_nb']] * opts['hidden_layer_nb'])
    model = torch.nn.DataParallel(model)
    checkpoint_epoch, score = load_checkpoint(model, checkpoint)
    return model


def load_checkpoints(model, resume_paths, strict=True):
    # Load models
    all_state_dicts = []
    all_epochs = []
    for resume_path in resume_paths:
        checkpoint = torch.load(resume_path)
        state_dict = checkpoint['state_dict']
        all_state_dicts.append(state_dict)
        all_epochs.append(checkpoint['epoch'])
    mean_state_dict = {}
    for state_key in state_dict.keys():
        if isinstance(state_dict[state_key], torch.cuda.LongTensor):
            mean_state_dict[state_key] = state_dict[state_key]
        else:
            params = [state_dict[state_key] for state_dict in all_state_dicts]
            mean_state_dict[state_key] = torch.stack(params).mean(0)
    
    model.load_state_dict(mean_state_dict, strict=strict)
    return max(all_epochs), None


def load_checkpoint(model,
                    resume_path,
                    optimizer=None,
                    strict=True,
                    load_atlas=False):
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        if 'module' in list(checkpoint['state_dict'].keys())[0]:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = {
                'module.{}'.format(key): item
                for key, item in checkpoint['state_dict'].items()
            }
            print("=> loaded checkpoint '{}' (epoch {})".format(
                resume_path, checkpoint['epoch']))

        missing_states = set(model.state_dict().keys()) - set(
            state_dict.keys())
        if len(missing_states) > 0:
            warnings.warn('Missing keys ! : {}'.format(missing_states))
        model.load_state_dict(state_dict, strict=strict)
        if optimizer is not None:
            try:
                missing_states = set(optimizer.state_dict().keys()) - set(
                    checkpoint['optimizer'].keys())
                if len(missing_states) > 0:
                    warnings.warn('Missing keys in optimizer ! : {}'.format(
                        missing_states))
                optimizer.load_state_dict(checkpoint['optimizer'])
            except ValueError:
                traceback.print_exc()
                warnings.warn(
                    'Couldn\' load optimizer from {}'.format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))
    best = checkpoint['score']
    return checkpoint['epoch'], best


def save_checkpoint(state,
                    is_best,
                    checkpoint='checkpoint',
                    filename='checkpoint.pth.tar',
                    snapshot=None):
    os.makedirs(checkpoint, exist_ok=True)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(checkpoint,
                         'checkpoint_{}.pth.tar'.format(state['epoch'])))

    if is_best:
        shutil.copyfile(filepath,
                        os.path.join(checkpoint, 'model_best.pth.tar'))
