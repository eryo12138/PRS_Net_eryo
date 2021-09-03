from PRS_Net import PRSNet
import torch
from m_dataloader import SymDataLoader
import scipy.io as sio
import ntpath
import os
import numpy as np


num_plane = 3
num_quat = 3
save_dir = './results/test1/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dataloader = SymDataLoader(phase="test")
dataset = dataloader.load_data()

model = torch.load("prs.pth", map_location=torch.device('cpu'))
print(model)

for i, data in enumerate(dataset):

    model.eval()
    with torch.no_grad():
        plane, quat = model(data['voxel'])

    data_path = data['path'][0]
    print('[%s] process mat ... %s' % (str(i), data_path))
    matdata = sio.loadmat(data_path, verify_compressed_data_integrity=False)

    short_path = ntpath.basename(data_path)
    name = os.path.splitext(short_path)[0]

    model = {'name': name, 'voxel': matdata['Volume'], 'vertices': matdata['vertices'], 'faces': matdata['faces'],
             'sample': np.transpose(matdata['surfaceSamples'])}
    for j in range(num_plane):
        model['plane' + str(j)] = plane[j].cpu().numpy()
    for j in range(num_quat):
        model['quat' + str(j)] = quat[j].cpu().numpy()

    sio.savemat(save_dir + "/" + name + ".mat", model)