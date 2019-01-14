import argparse
from copy import deepcopy
import os
import warnings

from matplotlib import pyplot as plt
import numpy as np
import torch
import trimesh

import argutils
from handobjectdatasets import cubes, shapenet, synthgrasps, shapedataset
from handobjectdatasets.queries import BaseQueries, TransQueries

from shapesdf.sdfnet import SFDNet
from shapesdf.netscripts import epochpass
from shapesdf.monitoring import Monitor, get_save_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset',
            type=str,
            default='synthgrasps',
            choices=[
                'cubes', 'shapenet',
                'synthgrasps', 'synthobjs'
                ])

    parser.add_argument('--use_cache', action='store_true', help='Use cache')
    parser.add_argument('--canonical', action='store_true', help='Use cache')
    parser.add_argument(
            '--mini_factor', type=float, default=0.01, help='Ratio in data to use (in ]0, 1[)')
    parser.add_argument(
            '--split', type=str, default='test', help='Usually [train|test]')
    parser.add_argument('--point_nb', type=int, default=10, help='point_nb^3 is the number of points sampled in the cube')
    parser.add_argument('--sdf_point_nb', type=int, default=200, help='Points to sample in the cube')
    parser.add_argument('--offset', type=int, default=0, help='point_nb^3 is the number of points sampled in the cube')

    # Model params
    parser.add_argument('--hidden_neuron_nb', type=int, default=64)
    parser.add_argument('--hidden_layer_nb', type=int, default=2)

    # Parallelization params
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=8)

    # Optimizer params
    parser.add_argument('--epoch_nb', type=int, default=1000)
    parser.add_argument('--optimizer', default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.1)

    # Saving params
    parser.add_argument('--res_folder', default='results')

    # Visualize params
    parser.add_argument('--display_freq', type=int, default=10)
    parser.add_argument('--epoch_display_freq', type=int, default=1)

    args = parser.parse_args()
    argutils.print_args(args)

    if args.dataset == 'shapenet':
        pose_dataset = shapenet.Shapenet(
                split=args.split,
                use_cache=args.use_cache,
                mini_factor=args.mini_factor,
                canonical=args.canonical)
    elif args.dataset == 'synthgrasps':
        pose_dataset = synthgrasps.SynthGrasps(
                split=args.split,
                use_cache=args.use_cache,
                root_palm=False,
                version=25,
                mini_factor=args.mini_factor,
                mode='obj',
                filter_class_ids=None,
                use_external_points=False)
    elif args.dataset == 'cubes':
        pose_dataset = cubes.Cubes(size=1000, mini_factor=args.mini_factor)

    model = SFDNet(inter_neurons=[args.hidden_neuron_nb] * args.hidden_layer_nb)
    queries = [TransQueries.objverts3d, BaseQueries.objfaces, TransQueries.sdf, TransQueries.sdf_points, TransQueries.objpoints3d]
    dataset = shapedataset.ShapeDataset(pose_dataset, queries=queries, sdf_point_nb=args.sdf_point_nb)
    train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=int(args.workers),
            drop_last=True)

    # Initialize optim tools
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    model.cuda()
    model.eval()
    fig = plt.figure()

    # Uniformly sample cube in [-1, 1]^3
    grid_point_nb = 40
    grid = np.mgrid[-1:1:complex(0, grid_point_nb), -1:1:complex(0, grid_point_nb), -1:1:complex(0, grid_point_nb)]
    uniform_grid = torch.Tensor(grid.reshape(3, -1)).unsqueeze(0).repeat(args.batch_size, 1, 1).cuda()

    # Prepare logging
    save_folder = get_save_folder(args.res_folder, args)
    if os.path.exists(save_folder):
        warnings.warn('Folder {} already exists!'.format(save_folder))
    else:
        print('Creating folder {}'.format(save_folder))
        os.makedirs(save_folder, exist_ok=True)


    hosting_folder = os.path.join(
            '/meleze/data0/public_html/yhasson/experiments/sdf_debug',
            save_folder)
    monitor = Monitor(save_folder, hosting_folder=hosting_folder)
    for epoch_idx in range(args.epoch_nb):
        train_avg_meters = epochpass.epoch_pass(train_loader, model, epoch_idx, optimizer=optimizer, train=True, fig=fig,
                save_folder=save_folder, vis_grid=uniform_grid, epoch_display_freq=args.epoch_display_freq, display_freq=args.display_freq)
        train_dict = {
                meter_name: meter.avg
                for meter_name, meter in
                train_avg_meters.average_meters.items()
                }
        monitor.log_train(epoch_idx + 1, train_dict)
        save_dict = {}
        for key in train_dict:
            save_dict[key] = {}
            save_dict[key]['train'] = train_dict[key]
        monitor.metrics.save_metrics(epoch_idx + 1, save_dict)
        monitor.metrics.plot_metrics()

