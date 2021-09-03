import os
import torch.utils.data as data
import scipy.io as sio
import torch


# 将代表每一个模型的文件的路径生成列表
def make_dataset(dir):

    data = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            data.append(path)
    return data


class SymDataset(data.Dataset):

    def __init__(self, root='./datasets/shapenet/', phase="train", batch_size=32):

        self.root = root
        self.phase = phase
        self.batchSize = batch_size

        # 训练数据的路径
        self.dir_train = os.path.join(self.root, self.phase)
        # 训练模型的文件路径列表
        self.train_paths = sorted(make_dataset(self.dir_train))
        # 总共有多少个模型
        self.dataset_size = len(self.train_paths)

    def __getitem__(self, index):

        data_path = self.train_paths[index]
        try:
            data = sio.loadmat(data_path, verify_compressed_data_integrity=False)
        except Exception as e:
            print(data_path, e)
            return None

        sample = data['surfaceSamples']
        voxel = data['Volume']
        cp = data['closestPoints']

        # 32*32*32->[1, 32, 32, 32]
        voxel = torch.from_numpy(voxel).float().unsqueeze(0)
        # 3*1000->1000*3
        sample = torch.from_numpy(sample).float().t()
        # (32, 32, 32, 3)->[32768, 3]
        cp = torch.from_numpy(cp).float().reshape(-1, 3)

        input_dict = {'voxel': voxel, 'sample': sample, 'cp': cp, 'path': data_path}

        return input_dict

    def __len__(self):
        # 使得不满一个batch的数据被舍弃
        # return self.dataset_size // self.batchSize * self.batchSize
        return self.dataset_size

    def name(self):
        return 'SymDataset'
