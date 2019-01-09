import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f

from handobjectdatasets.queries import BaseQueries, TransQueries

class SFDNet(nn.Module):
    def __init__(self,
                 atlas_ico_divisions=3,
                 bottleneck_size=512,
                 inter_neurons=[256, 256, 256],
                 dropout=0):
        super().__init__()
        if atlas_ico_divisions == 3:
            points_nb = 642

        # Initialize encoder
        self.bottleneck_size=bottleneck_size
        self.encoder = self.encoder = nn.Sequential(
            PointNetfeat(points_nb, global_feat=True, trans=False),
            nn.Linear(1024, self.bottleneck_size),
            # nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU())

        pred_neurons = [self.bottleneck_size] + inter_neurons

        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
                zip(pred_neurons[:-1], pred_neurons[1:])):
            if dropout:
                base_layers.append(nn.Dropout(p=dropout))
            if layer_idx == 0:
                # First layer receives shape features stacked with 3 coordinates 
                # inp_neurons = inp_neurons + 3 # TODO put back
                inp_neurons = 3
            base_layers.append(nn.Conv1d(inp_neurons, out_neurons, 1))
            base_layers.append(nn.ReLU())
        base_layers.append(nn.Conv1d(inter_neurons[-1], 1, 1))
        self.sdfpredictor = nn.Sequential(*base_layers)

    def forward(self,
                sample,
                no_loss=False,
                return_features=False,
                force_objects=False):
        inp_points3d = sample[TransQueries.objpoints3d]

        shape_features = self.encoder(inp_points3d)
        sampled_points = sample['sampled_points']
        features = shape_features.unsqueeze(2).repeat(1, 1, sampled_points.size(2))
        # TODO put back
        # stacked_features = torch.cat((sampled_points, features), 1)
        # pred_dists = self.sdfpredictor(stacked_features)
        pred_dists = self.sdfpredictor(sampled_points)
        return pred_dists


class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()
        self.identity = torch.from_numpy(
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)).view(
                1, 9).cuda()

    def forward(self, x):
        batch_size = x.size()[0]
        x = torch_f.relu(self.conv1(x))
        x = torch_f.relu(self.conv2(x))
        x = torch_f.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = torch_f.relu(self.fc1(x))
        x = torch_f.relu(self.fc2(x))
        x = self.fc3(x)

        identity = self.identity.repeat(batch_size, 1)
        x = x + identity
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=2500, global_feat=True, trans=False, feature_size=1024):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.feature_size = feature_size
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_size)
        self.trans = trans

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        x = torch_f.relu(self.bn1(self.conv1(x.transpose(1, 2))))
        pointfeat = x
        x = torch_f.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.feature_size)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, self.feature_size, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x
