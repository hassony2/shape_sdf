import argparse
from copy import deepcopy
import os
import warnings

from matplotlib import pyplot as plt
import numpy as np
import torch
import trimesh

from shapesdf.datasets import cubes, primitives, shapedataset
from shapesdf.datasets.queries import BaseQueries, TransQueries
from shapesdf.sdfnet import SFDNet
from shapesdf.netscripts import epochpass
from shapesdf.monitoring import Monitor, get_save_folder
from shapesdf import modelio
from shapesdf import argutils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset',
            type=str,
            default='primitives',
            choices=[
                'cubes', 'primitives',
                ])

    parser.add_argument('--use_cache', action='store_true', help='Use cache')
    parser.add_argument(
            '--mini_factor', type=float, default=0.01, help='Ratio in data to use (in ]0, 1[)')
    parser.add_argument('--sdf_point_nb', type=int, default=200, help='Points to sample in the cube')

    # Model params
    parser.add_argument('--hidden_neuron_nb', type=int, default=64, help='Number of hidden layers')
    parser.add_argument('--hidden_layer_nb', type=int, default=2, help='Number of hidden neurons in each hidden layer')

    # Parallelization params
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=8)

    # Optimizer params
    parser.add_argument('--epoch_nb', type=int, default=1000)
    parser.add_argument('--optimizer', default='sgd', choices=['adam', 'sgd'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.1, help='Momentum for SGD optimizer')

    # Saving params
    parser.add_argument('--save_folder_root', default='results')
    parser.add_argument('--pyapt_id', help='Internal id to keep track of running experiments')
    parser.add_argument('--snapshot', type=int, default=10)

    # Visualize params
    parser.add_argument('--display_freq', type=int, default=10, help='Batch display frequence')
    parser.add_argument('--epoch_display_freq', type=int, default=1, help='Epoch display frequence')

    args = parser.parse_args()
    argutils.print_args(args)

    # Prepare saving
    save_folder = get_save_folder(args.save_folder_root, args)
    if os.path.exists(save_folder):
        warnings.warn('Folder {} already exists!'.format(save_folder))
    else:
        print('Creating folder {}'.format(save_folder))
        os.makedirs(save_folder, exist_ok=True)

    argutils.save_args(args, save_folder, 'opt')

    if args.dataset == 'cubes':
        pose_dataset = cubes.Cubes(size=1000, mini_factor=args.mini_factor)
    elif args.dataset == 'primitives':
        pose_dataset = primitives.Primitives(size=1000, mini_factor=args.mini_factor)

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


    hosting_folder = None  # Path to folder where to save plotly graphs
    monitor = Monitor(save_folder, hosting_folder=hosting_folder)
    best_score = None
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

        # Get best score
        score = train_dict['loss']
        if best_score is None:
            is_best = True
        else:
            if new_score < best_score:
                is_best = True
            else:
                is_best = False
        modelio.save_checkpoint(
            {
                'epoch': epoch_idx + 1,
                'state_dict': model.cpu().state_dict(),
                'score': score,
                'optimizer': optimizer.state_dict(),
            },
            is_best=is_best,
            checkpoint=os.path.join(save_folder, 'checkpoints'),
            snapshot=args.snapshot)
        model.cuda()
