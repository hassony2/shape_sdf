import argparse
import os

from matplotlib import pyplot as plt
import numpy as np
import torch
import trimesh

import argutils
from handobjectdatasets import shapenet, synthgrasps, handataset
from handobjectdatasets.queries import BaseQueries, TransQueries

from shapesfd.sfdnet import SFDNet
from shapesfd.imgutils import plot_sdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--dataset',
            type=str,
            default='synthgrasps',
            choices=[
                'core50', 'fhbhands', 'ganhands', 'panoptic', 'shapenet',
                'synthgrasps', 'synthands', 'synthobjs', 'stereohands', 'tomasreal',
                'tzionas', 'yanademo', 'zimsynth'
                ])

    parser.add_argument('--use_cache', action='store_true', help='Use cache')
    parser.add_argument('--batch_size', type=int, default=2, help='Use cache')
    parser.add_argument('--canonical', action='store_true', help='Use cache')
    parser.add_argument(
            '--mini_factor', type=float, default=0.01, help='Ratio in data to use (in ]0, 1[)')
    parser.add_argument(
            '--split', type=str, default='test', help='Usually [train|test]')
    parser.add_argument('--point_nb', type=int, default=10, help='point_nb^3 is the number of points sampled in the cube')
    parser.add_argument('--offset', type=int, default=0, help='point_nb^3 is the number of points sampled in the cube')

    # Optimizer params
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.1)

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
                version=33,
                mini_factor=args.mini_factor,
                mode='obj',
                filter_class_ids=None)

        model = SFDNet()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.cuda()
    for idx in range(len(pose_dataset) - args.offset):
        verts, faces = pose_dataset.get_obj_verts_faces(idx + args.offset)
        print(verts.shape)
        # Render canonical
        centroid = verts.mean(0)
        centered_verts = verts - centroid
        max_radius = np.linalg.norm(centered_verts, 2, 1).max()
        canonical_verts = centered_verts / max_radius
        mesh = trimesh.Trimesh(vertices=canonical_verts,faces=faces)
        vertices_pt = torch.Tensor(canonical_verts).unsqueeze(0).repeat(args.batch_size, 1, 1).cuda()

        # Uniformly sample cube in [-1, 1]^3
        grid_point_nb = 10
        grid = np.mgrid[-1:1:complex(0, grid_point_nb), -1:1:complex(0, grid_point_nb), -1:1:complex(0, grid_point_nb)]
        uniform_grid = torch.Tensor(grid.reshape(3, -1)).unsqueeze(0).repeat(args.batch_size, 1, 1).cuda()
        # Randomly sample cube of in [-1, 1]^3
        fig = plt.figure()
        res_folder = 'results/sfd_1'
        for step_idx in range(1000):
            points = np.random.uniform(-1, 1, (args.point_nb, 3))
            distances = trimesh.proximity.signed_distance(mesh, grid.reshape(3, -1).transpose())
            dists_pt = torch.Tensor(distances).unsqueeze(0).repeat(args.batch_size, 1, 1).cuda()
            # distances = trimesh.proximity.signed_distance(mesh, points) # TODO put back
            # dists_pt = torch.Tensor(distances).unsqueeze(0).repeat(args.batch_size, 1, 1).cuda()
            points_pt = torch.Tensor(points).unsqueeze(0).repeat(args.batch_size, 1, 1).cuda()
            # sample = {TransQueries.objpoints3d: vertices_pt, 'sampled_points': points_pt.permute(0, 2, 1)} # TODO put back
            sample = {TransQueries.objpoints3d: vertices_pt, 'sampled_points': uniform_grid}

            pred_dists = model(sample)
            loss_val = torch.mean((pred_dists - dists_pt)**2)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            if step_idx % 5 == 0:
                cube_sample = {TransQueries.objpoints3d:vertices_pt, 'sampled_points': uniform_grid}
                cube_dists = model(cube_sample)
                cube_dists = cube_dists.cpu().detach()[0].view(grid_point_nb, grid_point_nb, grid_point_nb)
                dists_pt = dists_pt.cpu().detach()[0].view(grid_point_nb, grid_point_nb, grid_point_nb)
                plot_sdf(cube_dists, gt=dists_pt, grid_step=4)
                os.makedirs(res_folder, exist_ok=True)
                plt.savefig(
                        os.path.join(res_folder, '{}_{:06d}.png'.format(
                        idx, step_idx)))
                plt.clf()
                print('cube !')
                print('loss: {}, pred min: {} max: {}, gt min: {} max: {}'.format(loss_val.item(), cube_dists.min().item(), cube_dists.max().item(), dists_pt.min().item(), dists_pt.max().item()))
            print('loss: {}, pred min: {} max: {}, gt min: {} max: {}'.format(loss_val.item(), pred_dists.min().item(), pred_dists.max().item(), dists_pt.min().item(), dists_pt.max().item()))
