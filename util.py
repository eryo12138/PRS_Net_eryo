import torch
import numpy as np


def normalize(x, enddim=4):
    x  = x/(1E-12 + torch.norm(x[:,:enddim], dim=1, p=2, keepdim=True))
    return x

def planesymTransform(sample, plane):
    abc = plane[:,0:3].unsqueeze(1).repeat(1,sample.shape[1],1)
    d = plane[:,3].unsqueeze(1).unsqueeze(1).repeat(1,sample.shape[1],1)
    fenzi = torch.sum(sample*abc,2,True)+d
    fenmu = torch.norm(plane[:,0:3],2,1,True).unsqueeze(1).repeat(1,sample.shape[1],1)+1e-5
    x = 2*torch.div(fenzi,fenmu)
    y=torch.mul(x.repeat(1,1,3),abc/fenmu)
    return sample-y

def rotsymTransform(sample, quat):
    return rotate_module(sample, quat)

## points is Batch_size x P x 3,  #Bx4 quat vectors
def rotate_module(points, quat):
    nP = points.size(1)
    quat_rep = quat.unsqueeze(1).repeat(1, nP, 1)
    # print(quat_rep.shape)
    zero_points = 0 * points[:, :, 0].clone().view(-1, nP, 1)
    quat_points = torch.cat([zero_points, points], dim=2)

    rotated_points = quat_rot_module(quat_points, quat_rep)  # B x  P x 3
    return rotated_points

def quat_rot_module(points, quats):
    quatConjugate = quat_conjugate(quats)
    mult = hamilton_product(quats, points)
    mult = hamilton_product(mult, quatConjugate)
    return mult[:, :, 1:4]

def quat_conjugate(quat):
    # quat = quat.view(-1, 4)

    q0 = quat[:, :, 0]
    q1 = -1 * quat[:, :, 1]
    q2 = -1 * quat[:, :, 2]
    q3 = -1 * quat[:, :, 3]

    q_conj = torch.stack([q0, q1, q2, q3], dim=2)
    return q_conj

inds = torch.LongTensor([0, -1, -2, -3, 1, 0, 3, -2, 2, -3, 0, 1, 3, 2, -1, 0]).view(4, 4)

def hamilton_product(q1, q2):
    q_size = q1.size()
    # q1 = q1.view(-1, 4)
    # q2 = q2.view(-1, 4)
    q1_q2_prods = []
    for i in range(4):
        q2_permute_0 = q2[:, :, np.abs(inds[i][0])]
        q2_permute_0 = q2_permute_0 * np.sign(inds[i][0] + 0.01)

        q2_permute_1 = q2[:, :, np.abs(inds[i][1])]
        q2_permute_1 = q2_permute_1 * np.sign(inds[i][1] + 0.01)

        q2_permute_2 = q2[:, :, np.abs(inds[i][2])]
        q2_permute_2 = q2_permute_2 * np.sign(inds[i][2] + 0.01)

        q2_permute_3 = q2[:, :, np.abs(inds[i][3])]
        q2_permute_3 = q2_permute_3 * np.sign(inds[i][3] + 0.01)
        q2_permute = torch.stack([q2_permute_0, q2_permute_1, q2_permute_2, q2_permute_3], dim=2)

        q1q2_v1 = torch.sum(q1 * q2_permute, dim=2, keepdim=True)
        q1_q2_prods.append(q1q2_v1)
    # print(q1_q2_prods[0].shape)
    q_ham = torch.cat(q1_q2_prods, dim=2)
    # q_ham = q_ham.view(q_size)
    return q_ham


class calDistence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, trans_points, cp, voxel, gridSize, weight=1):
        nb = pointClosestCellIndex(trans_points)
        idx = torch.matmul(nb, torch.cuda.FloatTensor([gridSize ** 2, gridSize, 1])).long()
        mask = 1 - torch.gather(voxel.view(-1, gridSize ** 3), 1, idx)
        idx = idx.unsqueeze(2)
        idx = idx.repeat(1, 1, 3)
        mask = mask.unsqueeze(2).repeat(1, 1, 3)
        closest_points = torch.gather(cp, 1, idx)
        ctx.constant = weight
        distance = trans_points - closest_points
        distance = distance * mask
        ctx.save_for_backward(distance)
        return torch.mean(torch.sum(torch.sum(torch.pow(distance, 2), 2), 1)) * weight

    @staticmethod
    def backward(ctx, grad_output):
        distance = ctx.saved_tensors
        distance = distance[0]
        grad_trans_points = 2 * (distance) * ctx.constant / (distance.shape[0])
        return grad_trans_points, None, None, None, None

def pointClosestCellIndex(points, gridBound = 0.5, gridSize = 32):
    gridMin = -gridBound + gridBound / gridSize
    gridMax = gridBound - gridBound / gridSize
    inds = (points - gridMin) * gridSize / (2 * gridBound)
    inds = torch.round(torch.clamp(inds, min=0, max=gridSize-1))
    return inds