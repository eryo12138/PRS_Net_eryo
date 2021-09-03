from torch import nn
import torch
from util import planesymTransform, rotsymTransform, calDistence, normalize


class RegularLoss(nn.Module):
    def __init__(self):
        super(RegularLoss, self).__init__()
        self.eye = torch.eye(3).cuda()

    def __call__(self, plane=None, quat=None, weight=1):

        reg_rot = torch.Tensor([0]).cuda()
        reg_plane = torch.Tensor([0]).cuda()
        if plane:
            # [2, 3]->[2, 3, 1]
            p = [normalize(i[:, 0:3]).unsqueeze(2) for i in plane]

            # [2, 3, 4]
            x = torch.cat(p, 2)

            y = torch.transpose(x, 1, 2)
            # [2, 3, 3]
            reg_plane = (torch.matmul(x, y) - self.eye).pow(2).sum(2).sum(1).mean() * weight
        if quat:
            q = [i[:, 1:4].unsqueeze(2) for i in quat]
            x = torch.cat(q, 2)
            y = torch.transpose(x, 1, 2)
            reg_rot = (torch.matmul(x, y) - self.eye).pow(2).sum(2).sum(1).mean() * weight
        return reg_plane, reg_rot


class symLoss(nn.Module):
    def __init__(self, gridBound, gridSize):
        super(symLoss, self).__init__()
        self.gridSize = gridSize
        self.gridBound = gridBound
        self.cal_distance = calDistence.apply

    def __call__(self, points, cp, voxel, plane=None, quat=None, weight=1):
        ref_loss = torch.Tensor([0]).cuda()
        rot_loss = torch.Tensor([0]).cuda()
        for p in plane:
            ref_points = planesymTransform(points, p)
            ref_loss += self.cal_distance(ref_points, cp, voxel, self.gridSize)
        for q in quat:
            rot_points = rotsymTransform(points, q)
            rot_loss += self.cal_distance(rot_points, cp, voxel, self.gridSize)
        return ref_loss, rot_loss