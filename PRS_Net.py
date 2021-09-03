import torch
import torch.nn as nn
from util import normalize

# 总的网络结构
class PRSNet(nn.Module):
    def __init__(self, num_plane, num_quat, biasTerms):
        super(PRSNet, self).__init__()

        self.num_plane = num_plane
        self.num_quat = num_quat
        # 初始化的偏置值
        self.biasTerm = biasTerms

        self.featureFetch = FeatureFetch().cuda()
        self.symFc = Sym_FC(num_plane = self.num_plane, num_quat=self.num_quat, biasTerms=self.biasTerm).cuda()

    def forward(self, voxel):
        return self.symFc(self.featureFetch(voxel))


# 卷积层，提取特征[2, 64, 1, 1, 1]
class FeatureFetch(nn.Module):
    def __init__(self):
        super(FeatureFetch, self).__init__()

        self.pooling = nn.MaxPool3d(kernel_size=2)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.model = torch.nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=4),
            self.pooling,
            self.activation,

            nn.Conv3d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=8),
            self.pooling,
            self.activation,

            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=16),
            self.pooling,
            self.activation,

            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=32),
            self.pooling,
            self.activation,

            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(num_features=64),
            self.pooling,
            self.activation,
        )

    def forward(self, input):
        return self.model(input)


# 得到对称平面和旋转轴
class Sym_FC(nn.Module):
    def __init__(self, num_plane, num_quat, biasTerms):
        super(Sym_FC, self).__init__()

        self.num_plane = num_plane
        self.num_quat = num_quat

        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        for i in range(self.num_plane):
            last = nn.Linear(in_features=16, out_features=4)
            last.bias.data = torch.Tensor(biasTerms['plane' + str(i+1)])
            model = nn.Sequential(
                nn.Linear(in_features=64, out_features=32),
                self.activation,
                nn.Linear(in_features=32, out_features=16),
                self.activation,
                last,
                self.activation,
            )
            setattr(self, 'planeLayer' + str(i + 1), model)

        for i in range(self.num_quat):
            last = nn.Linear(in_features=16, out_features=4)
            last.bias.data = torch.Tensor(biasTerms['quat' + str(i+1)])
            model = nn.Sequential(
                nn.Linear(in_features=64, out_features=32),
                self.activation,
                nn.Linear(in_features=32, out_features=16),
                self.activation,
                last,
                self.activation,
            )
            setattr(self, 'quatLayer' + str(i + 1), model)

    def forward(self, feature):
        feature = feature.view(feature.size(0), -1)
        quat = []
        plane = []
        for i in range(self.num_quat):
            quatLayer = getattr(self, 'quatLayer' + str(i + 1))
            quat += [normalize(quatLayer(feature))]

        for i in range(self.num_plane):
            planeLayer = getattr(self, 'planeLayer' + str(i + 1))
            plane += [normalize(planeLayer(feature), 3)]
        return quat, plane
