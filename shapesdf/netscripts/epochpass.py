from copy import deepcopy
import os

import numpy as np
import torch
from tqdm import tqdm

from shapesdf.datasets.queries import BaseQueries, TransQueries
from shapesdf.imgutils import plot_sdf, visualize_sample
from shapesdf.evalutils import AverageMeters

def epoch_pass(loader, model, epoch_idx, optimizer=None, train=True, fig=None, save_folder=None, vis_grid=None, display_freq=10, epoch_display_freq=1):
    avg_meters = AverageMeters()

    img_folder = os.path.join(save_folder, 'images')
    os.makedirs(img_folder, exist_ok=True)
    for sample_idx, sample in enumerate(tqdm(loader)):
        results, loss_val = model(sample)
        if train:
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()


        if sample_idx % display_freq == 0 and epoch_idx % epoch_display_freq == 0:
            sample_vis = deepcopy(sample)
            sample_vis[TransQueries.sdf_points] = vis_grid.transpose(2, 1)
            results_vis, _ = model(sample_vis, no_loss=True)
            visualize_sample(sample, results_vis, fig)
            save_path = os.path.join(img_folder, '{:06d}_{:06d}.png'.format(
                epoch_idx, sample_idx))
            fig.savefig(save_path, bbox_inches='tight', dpi=190)
            print('Saving sample to {}'.format(save_path))
            fig.clf()
        avg_meters.add_loss_value('loss', loss_val.item())
    return avg_meters
