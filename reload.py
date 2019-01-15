import argparse
from copy import deepcopy
import os
import warnings

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes_lewiner
import torch
import trimesh
from tqdm import tqdm

import argutils
from handobjectdatasets import cubes, primitives, shapenet, synthgrasps, shapedataset
from handobjectdatasets.queries import BaseQueries, TransQueries

from shapesdf.sdfnet import SFDNet
from shapesdf.netscripts import epochpass
from shapesdf.monitoring import Monitor
from shapesdf import modelio
from shapesdf.imgutils import plot_sdf, visualize_sample

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset',
            type=str,
            default='synthgrasps',
            choices=[
                'cubes', 'primitives', 'shapenet',
                'synthgrasps', 'synthobjs'
                ])

    parser.add_argument('--sdf_point_nb', type=int, default=200, help='Points to sample in the cube')
    parser.add_argument('--grid_point_nb', type=int, default=40, help='Number of points on one side of voxel grid')
    parser.add_argument('--use_cache', action='store_true', help='Use cache')
    parser.add_argument('--canonical', action='store_true', help='Use cache')
    parser.add_argument(
            '--mini_factor', type=float, default=0.01, help='Ratio in data to use (in ]0, 1[)')
    parser.add_argument(
            '--split', type=str, default='test', help='Usually [train|test]')
    parser.add_argument('--offset', type=int, default=0, help='point_nb^3 is the number of points sampled in the cube')

    # Parallelization params
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)

    # Saving params
    parser.add_argument('--checkpoint', default='results')

    args = parser.parse_args()
    argutils.print_args(args)

    if args.dataset == 'shapenet':
        pose_dataset = shapenet.Shapenet(
                split=args.train_split,
                use_cache=args.use_cache,
                mini_factor=args.mini_factor,
                canonical=args.canonical)
    elif args.dataset == 'synthgrasps':
        pose_dataset = synthgrasps.SynthGrasps(
                split=args.train_split,
                use_cache=args.use_cache,
                root_palm=False,
                version=25,
                mini_factor=args.mini_factor,
                mode='obj',
                filter_class_ids=None,
                use_external_points=False)
    elif args.dataset == 'cubes':
        pose_dataset = cubes.Cubes(size=1000, mini_factor=args.mini_factor)
    elif args.dataset == 'primitives':
        pose_dataset = primitives.Primitives(size=1000, mini_factor=args.mini_factor)

    model = modelio.reload(args.checkpoint)
    queries = [TransQueries.objverts3d, BaseQueries.objfaces, TransQueries.sdf, TransQueries.sdf_points, TransQueries.objpoints3d]
    dataset = shapedataset.ShapeDataset(pose_dataset, queries=queries, sdf_point_nb=args.sdf_point_nb)
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            drop_last=True)

    model.cuda()
    model.eval()

    # Uniformly sample cube in [-1, 1]^3
    grid_point_nb = args.grid_point_nb
    extreme_val = 1.2
    grid = np.mgrid[-extreme_val:extreme_val:complex(0, grid_point_nb), -extreme_val:extreme_val:complex(0, grid_point_nb), -extreme_val:extreme_val:complex(0, grid_point_nb)]
    uniform_grid = torch.Tensor(grid.reshape(3, -1)).unsqueeze(0).repeat(args.batch_size, 1, 1).cuda()

    for sample_idx, sample in enumerate(tqdm(loader)):
        sample[TransQueries.sdf_points] = uniform_grid.transpose(2, 1)
        results, loss_val = model(sample, no_loss=True)

        # Get cube predictions
        preds = results['pred_dists'].detach().cpu()
        side_size = int(round(np.power(preds.shape[1], 1/3)))
        preds = preds.reshape(preds.shape[0], side_size, side_size, side_size)

        fig = plt.figure(figsize=(10, 10))
        visualize_sample(sample, results, fig)
        plt.show()

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        verts, faces, normals, values = marching_cubes_lewiner(preds[0].numpy(), level=0)
        mesh = Poly3DCollection(verts[faces])
        mesh.set_edgecolor([0.5, 0.5, 0.5])
        mesh.set_facecolor([0.6, 0.8, 1, 0.2])
        ax.add_collection3d(mesh)
        # ax.voxels(preds[0].numpy() > 0, edgecolor='k')

        plt.tight_layout()
        ax.set_xlim(-0, 40)
        ax.set_ylim(-0, 40)
        ax.set_zlim(-0, 40)
        plt.show()

