from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

from shapesdf.datasets.queries import BaseQueries, TransQueries

def signed_dist_img(sdf_points, sdf_vals, point_min=-1, point_max=1, cube_resolution=40):
    """
    Args:
        sdf_points: points at which sdf is known (batch_sizexpoint_nbx3)
        sdf_vals: sdf values (batch_sizexpoint_nb)
    """
    point_span = 2
    sdf_points_scaled = (sdf_points - point_min) / (point_max - point_min)
    batch_idxs = (sdf_points_scaled * (40 - 1)).long()
    cube = sdf_points.new_zeros((sdf_points.shape[0], cube_resolution, cube_resolution, cube_resolution))
    for batch_idx in range(sdf_points.shape[0]):
        batch_cube = cube[batch_idx]
        for idx, val in zip(batch_idxs[batch_idx], sdf_vals[batch_idx]):
            for i in range(-point_span, point_span):
                for j in range(-point_span, point_span):
                    for k in range(-point_span, point_span):
                        if (idx[0] + i in list(range(cube_resolution))) and (idx[1] + j in list(range(cube_resolution))) and (idx[2] + k in list(range(cube_resolution))):
                            batch_cube[idx[0] + i, idx[1] + j, idx[2] + k] = val
    return cube

def visualize_sample(sample, results, fig=None, grid_step=8, max_samples=2, cmap='seismic'):
    if fig is None:
        fig = plt.figure(figsize=(3, 2))
    preds = results['pred_dists'].detach().cpu()
    side_size = int(round(np.power(preds.shape[1], 1/3)))

    # Reshape prediction into cube !
    preds = preds.view(preds.shape[0], side_size, side_size, side_size)
    col_nb = int(preds.shape[1] / grid_step) + 1  # scatter (1) + grid

    input_points = sample[TransQueries.objpoints3d]
    sdf_vals = sample[TransQueries.sdf]
    sdf_points = sample[TransQueries.sdf_points]
    signed_cube = signed_dist_img(sdf_points, sdf_vals)

    sample_nb = min(max_samples, preds.shape[0])
    row_nb = sample_nb * 2

    for sample_idx in range(sample_nb):
        ax = fig.add_subplot(2*sample_nb, col_nb, (2 * sample_idx) * col_nb + 1)
        batch_points = input_points[sample_idx]
        ax.scatter(batch_points[:, 1], batch_points[:, 2], alpha=0.2, s=2)
        for col_idx in range(1, col_nb):
            grid_idx = (col_idx - 1) * grid_step
            ax = fig.add_subplot(2*sample_nb, col_nb, 2 * sample_idx * col_nb + col_idx + 1)
            cx = ax.imshow(preds[sample_idx][grid_idx], vmin=-1, vmax=1, cmap=cmap)
            fig.colorbar(cx, ax=ax, ticks=[-1, 0, 1])
            ax.axis('off')
            ax = fig.add_subplot(2*sample_nb, col_nb, (1 + 2 * sample_idx) * col_nb + col_idx + 1)
            cx = ax.imshow(signed_cube[sample_idx][grid_idx], vmin=-1, vmax=1, cmap=cmap)
            fig.colorbar(cx, ax=ax, ticks=[-1, 0, 1])
            ax.axis('off')


def plot_sdf(grid_vals, fig=None, gt=None, grid_step=1, cmap='seismic'):
    if fig is None:
        fig = plt.figure(figsize=(3, 2))
    plot_nb = int(grid_vals.shape[0] / grid_step)

    if gt is None:
        row_nb = 1
    else:
        row_nb = 2
    for plot_idx in range(0, plot_nb):
        ax = fig.add_subplot(row_nb, plot_nb, plot_idx + 1)
        plt_im = ax.imshow(grid_vals[grid_step * plot_idx], vmin=-2, vmax=1, cmap=cmap)
        fig.colorbar(plt_im)

    if gt is not None:
        for plot_idx in range(0, plot_nb):
            plt.subplot(row_nb, plot_nb, plot_nb + plot_idx + 1)
            ax = plt.imshow(gt[grid_step * plot_idx], vmin=-2, vmax=1, cmap=cmap)
            plt.colorbar(ax)
