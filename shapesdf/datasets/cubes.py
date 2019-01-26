import json
import os
import pickle
import random
import warnings

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFile
from scipy.stats import special_ortho_group  
import scipy
import trimesh
from tqdm import tqdm

from shapesdf.datasets.queries import (BaseQueries, TransQueries,
        get_trans_queries)
from shapesdf.datasets import vertexsample
from shapesdf.datasets.objutils import fast_load_obj

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Cubes():
    def __init__(
            self,
            size=1000,
            split='train',
            mini_factor=1):
        """
        Dataset of cubes randomly rotated around their center
        """
        self.split = split
        self.size = int(size * mini_factor)
        self.all_queries = [BaseQueries.objverts3d, BaseQueries.objfaces, BaseQueries.objpoints3d]
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        self.name = 'cubes'
        self.mini_factor = mini_factor
        self.cam_intr = np.array([[480., 0., 128.], [0., 480., 128.],
            [0., 0., 1.]]).astype(np.float32)
        self.cam_extr = np.array([[1., 0., 0., 0.], [0., -1., 0., 0.],
            [0., 0., -1., 0.]]).astype(np.float32)
        self.load_dataset()


    def load_dataset(self):
        all_vertices = []
        _, faces = _create_cube()
        self.faces = faces
        for sample_idx in range(self.size):
            vertices, _ = _create_cube()
            all_vertices.append(vertices)
        self.objverts3d = all_vertices

    def get_obj_verts_faces(self, idx):
        faces = self.faces
        vertices = self.objverts3d[idx]
        return vertices, faces

    def get_objpoints3d(self, idx, point_nb=600):
        points = vertexsample.points_from_mesh(self.faces, self.objverts3d[idx], vertex_nb=point_nb)
        return points

    def __len__(self):
        return self.size

def _create_cube(center=True, random_rot=True):
    vertices = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,1,1], [1,1,1], [1,0,1], [0,0,1]])
    faces = np.array([[0, 2, 1], [0, 3, 2], [2, 3, 4], [2, 4, 5], [1,2,5], [1,5,6], [0, 7,4], [0, 4, 3], [5, 4, 7], [5, 7, 6], [0, 6, 7], [0, 1, 6]])
    if center:
        vertices = vertices - 0.5
    if random_rot:
        rot_mat = special_ortho_group.rvs(3)
        vertices = rot_mat.dot(vertices.transpose()).transpose()
    return vertices, faces
