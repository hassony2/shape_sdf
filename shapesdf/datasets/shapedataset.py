import numbers
import os
import pickle
import random
import traceback

import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import ImageStat, Image, ImageFilter
import torch
from torch.utils.data import Dataset
import trimesh

from shapesdf.datasets import transformutils
from shapesdf.datasets.queries import BaseQueries, TransQueries, one_query_in, no_query_in


class ShapeDataset(Dataset):
    """Class inherited by hands datasets
    hands datasets must implement the following methods:
    - get_image
    that respectively return a PIL image and a numpy array
    - the __len__ method

    and expose the following attributes:
    - the cache_folder : the path to the cache folder of the dataset
    """

    def __init__(self,
                 pose_dataset,
                 point_nb=600,
                 sdf_point_nb=200,
                 inp_res=256,
                 max_rot=np.pi,
                 normalize_img=False,
                 split='train',
                 scale_jittering=0.3,
                 center_jittering=0.2,
                 train=True,
                 hue=0.15,
                 saturation=0.5,
                 contrast=0.5,
                 brightness=0.5,
                 blur_radius=0.5,
                 queries=[
                     BaseQueries.images,
                     TransQueries.verts3d,
                 ],
                 block_rot=False,
                 rot3d=True):
        # Dataset attributes
        self.rot3d = rot3d  # Disables 3d rotation to match camera view
        self.pose_dataset = pose_dataset
        self.inp_res = inp_res
        self.point_nb = point_nb
        self.sdf_point_nb = sdf_point_nb
        self.normalize_img = normalize_img

        # Color jitter attributes
        self.hue = hue
        self.contrast = contrast
        self.brightness = brightness
        self.saturation = saturation
        self.blur_radius = blur_radius

        self.max_rot = max_rot
        self.block_rot = block_rot

        # Training attributes
        self.train = train
        self.scale_jittering = scale_jittering
        self.center_jittering = center_jittering

        self.queries = queries

    def __len__(self):
        return len(self.pose_dataset)

    def get_sample(self, idx, query=None):
        if query is None:
            query = self.queries
        sample = {}

        if BaseQueries.images in query or TransQueries.images in query:
            center, scale = self.pose_dataset.get_center_scale(idx)
            needs_center_scale = True
        else:
            needs_center_scale = False

        flip = False
        # Get original image
        if BaseQueries.images in query or TransQueries.images in query:
            img = self.pose_dataset.get_image(idx)
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if BaseQueries.images in query:
                sample[BaseQueries.images] = img

        # Get depth image
        if BaseQueries.depth in query or TransQueries.depth in query:
            depth = self.pose_dataset.get_depth(idx)
            if flip:
                depth = depth[:, ::-1]
            if BaseQueries.depth in query:
                sample[BaseQueries.depth] = depth

        # Flip and image 2d if needed
        if flip:
            center[0] = img.size[0] - center[0]
        # Data augmentation
        if self.train and needs_center_scale:
            # Randomly jitter center
            # Center is located in square of size 2*center_jitter_factor
            # in center of cropped image
            center_offsets = self.center_jittering * scale * np.random.uniform(
                low=-1, high=1, size=2)
            center = center + center_offsets.astype(int)

            # Scale jittering
            scale_jittering = self.scale_jittering * np.random.randn() + 1
            scale_jittering = np.clip(scale_jittering,
                                      1 - self.scale_jittering,
                                      1 + self.scale_jittering)
            scale = scale * scale_jittering

            rot = 2 * self.max_rot * torch.rand(1).item() - self.max_rot
        else:
            rot = 0
            if self.block_rot:
                rot = self.max_rot
        rot_mat = np.array([[np.cos(rot), -np.sin(rot),
                             0], [np.sin(rot), np.cos(rot), 0],
                            [0, 0, 1]]).astype(np.float32)

        # Get 2D hand joints
        if (TransQueries.images in query) or TransQueries.depth in query:
            affinetrans, post_rot_trans = transformutils.get_affine_transform(
                center, scale, [self.inp_res, self.inp_res], rot=rot)
            if TransQueries.affinetrans in query:
                sample[TransQueries.affinetrans] = torch.from_numpy(
                    affinetrans)

        if BaseQueries.camintrs in query or TransQueries.camintrs in query:
            camintr = self.pose_dataset.get_camintr(idx)
            if BaseQueries.camintrs in query:
                sample[BaseQueries.camintrs] = camintr
            if TransQueries.camintrs in query:
                # Rotation is applied as extr transform
                new_camintr = post_rot_trans.dot(camintr)
                sample[TransQueries.camintrs] = new_camintr

        # Get 2D object points
        if BaseQueries.objpoints2d in query or TransQueries.objpoints2d in query:
            objpoints2d = self.pose_dataset.get_objpoints2d(idx)
            if flip:
                objpoints2d = objpoints2d.copy()
                objpoints2d[:, 0] = img.size[0] - objpoints2d[:, 0]
            if BaseQueries.objpoints2d in query:
                sample[BaseQueries.objpoints2d] = torch.from_numpy(objpoints2d)
            if TransQueries.objpoints2d in query:
                transobjpoints2d = transformutils.transform_coords(
                    objpoints2d, affinetrans)
                sample[TransQueries.objpoints2d] = torch.from_numpy(
                    np.array(transobjpoints2d))

        # Get segmentation
        if BaseQueries.segms in query or TransQueries.segms in query:
            segm = self.pose_dataset.get_segm(idx)
            if flip:
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)
            if BaseQueries.segms in query:
                sample[BaseQueries.segms] = segm
            if TransQueries.segms in query:
                segm = transformutils.transform_img(segm, affinetrans,
                                               [self.inp_res, self.inp_res])
                segm = segm.crop((0, 0, self.inp_res, self.inp_res))
                segm = func_transforms.to_tensor(segm)
                sample[TransQueries.segms] = segm

        # Get 3D object points
        if (TransQueries.objverts3d in query) and (
                  BaseQueries.objverts3d in self.pose_dataset.all_queries):
            obj_verts3d, obj_faces = self.pose_dataset.get_obj_verts_faces(idx)
            if flip:
                obj_verts3d[:, 0] = -obj_verts3d[:, 0]
            if BaseQueries.objverts3d in query:
                sample[BaseQueries.objverts3d] = obj_verts3d
            if TransQueries.objverts3d in query:
                if self.rot3d: 
                    origin_trans_mesh = rot_mat.dot(
                        obj_verts3d.transpose(1, 0)).transpose()
                else:
                    origin_trans_mesh = obj_verts3d
                # Normalize vertices
                center3d = (origin_trans_mesh.max(0) + origin_trans_mesh.min(0)) / 2
                centered_verts = origin_trans_mesh - center3d
                max_radius = np.linalg.norm(centered_verts, 2, 1).max()
                canonical_verts = centered_verts / max_radius
                # sample[TransQueries.objverts3d] = canonical_verts

            # if BaseQueries.objfaces in query:
            #     sample[BaseQueries.objfaces] = obj_faces

        if TransQueries.sdf in query:
            # Sample points in unit cube
            uniform_points = 2 * torch.rand(self.sdf_point_nb, 3) - 1
            mesh = trimesh.Trimesh(vertices=canonical_verts,faces=obj_faces)
            distances = trimesh.proximity.signed_distance(mesh, uniform_points)
            sample[TransQueries.sdf] = distances
            sample[TransQueries.sdf_points] = uniform_points

        # Get 3D object points
        if TransQueries.objpoints3d in query and (
                BaseQueries.objpoints3d in self.pose_dataset.all_queries):
            points3d = self.pose_dataset.get_objpoints3d(
                idx, point_nb=self.point_nb)
            if flip:
                points3d[:, 0] = -points3d[:, 0]
            if self.rot3d:
                points3d = rot_mat.dot(points3d.transpose(1, 0)).transpose()
            # Normalize points
            center3d = (points3d.max(0) + points3d.min(0)) / 2
            centered_points = points3d - center3d
            max_radius = np.linalg.norm(centered_points, 2, 1).max()
            canonical_points = centered_points / max_radius
            sample[TransQueries.objpoints3d] = canonical_points

        if TransQueries.center3d in query:
            sample[TransQueries.center3d] = center3d

        # Get rgb image
        if TransQueries.images in query:
            # Data augmentation
            if self.train:
                blur_radius = torch.rand(1).item() * self.blur_radius
                img = img.filter(ImageFilter.GaussianBlur(blur_radius))
                img = imgtrans.color_jitter(
                    img,
                    brightness=self.brightness,
                    saturation=self.saturation,
                    hue=self.hue,
                    contrast=self.contrast)
            # Transform and crop
            img = transformutils.transform_img(img, affinetrans,
                                          [self.inp_res, self.inp_res])
            img = img.crop((0, 0, self.inp_res, self.inp_res))

            # Tensorize and normalize_img
            img = func_transforms.to_tensor(img).float()
            if self.black_padding:
                padding_ratio = 0.2
                padding_size = int(self.inp_res * padding_ratio)
                img[:, 0:padding_size, :] = 0
                img[:, -padding_size:-1, :] = 0
                img[:, :, 0:padding_size] = 0
                img[:, :, -padding_size:-1] = 0

            if self.normalize_img:
                img = func_transforms.normalize(img, self.mean, self.std)
            else:
                img = func_transforms.normalize(img, [0.5, 0.5, 0.5],
                                                [1, 1, 1])
            if TransQueries.images in query:
                sample[TransQueries.images] = img

        # Get transformed depth image
        if TransQueries.depth in query:
            # Transform and crop
            depth = cv2.warpAffine(depth, affinetrans[:2], (self.inp_res,
                                                            self.inp_res))

            # Tensorize depth
            if TransQueries.depth in query:
                sample[TransQueries.depth] = depth.astype(np.float32)

        # Add meta information
        if BaseQueries.meta in query:
            meta = self.pose_dataset.get_meta(idx)
            sample[BaseQueries.meta] = meta
        return sample

    def __getitem__(self, idx):
        try:
            sample = self.get_sample(idx, self.queries)
        except Exception:
            traceback.print_exc()
            print('Encountered error processing sample {}'.format(idx))
            random_idx = torch.randint(low=0, high=len(self), size=(1, 1)).item()
            sample = self.get_sample(random_idx, self.queries)
        return sample


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
