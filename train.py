from PRS_Net import PRSNet
import numpy as np

from loss import symLoss, RegularLoss
from m_dataloader import SymDataLoader
import time
import torch
from torch.autograd import Variable


###########################网络#########################
num_plane = 4
num_quat = 4
biasTerms={}
biasTerms['plane1']=[1,0,0,0]
biasTerms['plane2']=[0,1,0,0]
biasTerms['plane3']=[0,0,1,0]
biasTerms['quat1']=[0, 0, 0, np.sin(np.pi/2)]
biasTerms['quat2']=[0, 0, np.sin(np.pi/2), 0]
biasTerms['quat3']=[0, np.sin(np.pi/2), 0, 0]
if num_plane > 3:
    for i in range(4, num_plane+1):
        plane = np.random.random_sample((3,))
        biasTerms['plane'+str(i)] = (plane/np.linalg.norm(plane)).tolist()+[0]
if num_quat > 3:
    for i in range(4, num_quat+1):
        quat = np.random.random_sample((4,))
        biasTerms['quat'+str(i)] = (quat/np.linalg.norm(quat)).tolist()

PRSNet = PRSNet(num_plane=num_plane, num_quat=num_quat, biasTerms=biasTerms)
print(PRSNet)


#########################数据##############################
dataloader = SymDataLoader()
dataset = dataloader.load_data()
dataset_size = len(dataloader)
print("#training images = %d" % dataset_size)


#########################损失函数###########################
weight = 25     # 误差权重
gridBound = 0.5
gridSize = 32
sym_loss = symLoss(gridBound, gridSize)
reg_loss = RegularLoss()

#########################优化器#############################
lr = 0.001
beta = (0.9, 0.999)
optimizer = torch.optim.Adam(PRSNet.parameters(), lr=lr, betas = beta)


#########################训练##############################
epoch = 10
for e in range(epoch):
    epoch_start_time = time.time()

    for i, data in enumerate(dataset):

        ###forward###
        # ([2, 1, 32, 32, 32])
        voxel = data['voxel']
        # ([2, 1000, 3])
        sample = data['sample']
        # ([2, 32768, 3])
        cp = data['cp']

        voxel = Variable(voxel.data.cuda(), requires_grad=True)
        points = Variable(sample.data.cuda())
        cp = Variable(cp.data.cuda())

        quat, plane = PRSNet(voxel)

        loss_ref, loss_rot = sym_loss(points, cp, voxel, plane=plane, quat=quat)
        loss_reg_plane, loss_reg_rot = reg_loss(plane=plane, quat=quat, weight=weight)

        losses = [loss_ref, loss_rot,loss_reg_plane, loss_reg_rot]
        losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
        loss_names = ['ref', 'rot', 'reg_plane', 'reg_rot']
        losses_dict = dict(zip(loss_names, losses))
        loss = (losses_dict['ref'] + losses_dict['rot'] + losses_dict['reg_plane'] + losses_dict['reg_rot'])

        ###backward###
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (e, epoch, epoch_end_time - epoch_start_time))

torch.save(PRSNet, "./prs.pth")
print("模型已保存")
