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


class Primitives():
    def __init__(
            self,
            size=1000,
            split='train',
            mini_factor=1,
            shape_folder='datasets/primitives'):
        """
        Dataset of blender primitices (cube, torus, cylinder, sphere, ...)
        """
        self.split = split
        self.size = int(size * mini_factor)
        self.all_queries = [BaseQueries.objverts3d, BaseQueries.objfaces, BaseQueries.objpoints3d]
        trans_queries = get_trans_queries(self.all_queries)
        self.all_queries.extend(trans_queries)

        self.name = 'primitives'
        self.shape_folder = shape_folder
        self.mini_factor = mini_factor

        self.load_dataset()

    def load_dataset(self):
        shape_names = [shape_name for shape_name in os.listdir(self.shape_folder) if '.obj' in shape_name]
        meshes = []
        for shape_name in shape_names:
            shape_path = os.path.join(self.shape_folder,  shape_name)
            shape_mesh = trimesh.load(shape_path)
            assert shape_mesh.is_watertight, 'mesh at {} should be watertight'.format(shape_path)
            meshes.append(shape_mesh)

        objverts3d = []
        objfaces = []
        for sample_idx in range(self.size):
            mesh = random.choice(meshes)
            vertices = np.array(mesh.vertices)
            objfaces.append(np.array(mesh.faces))
            rot_mat = special_ortho_group.rvs(3)
            vertices = rot_mat.dot(vertices.transpose()).transpose()
            objverts3d.append(vertices)
        self.objverts3d = objverts3d
        self.objfaces = objfaces

    def get_obj_verts_faces(self, idx):
        faces = self.objfaces[idx]
        vertices = self.objverts3d[idx]
        return vertices, faces

    def get_objpoints3d(self, idx, point_nb=600):
        points = vertexsample.points_from_mesh(self.objfaces[idx], self.objverts3d[idx], vertex_nb=point_nb, show_cloud=False)
        return points

    def __len__(self):
        return self.size
