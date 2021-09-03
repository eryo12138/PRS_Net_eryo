from m_dataset import SymDataset
import torch


class SymDataLoader():

    def __init__(self, root='./datasets/shapenet/', phase="train", batch_size=32, shuffle=False):

        self.root = root
        self.phase = phase
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset = SymDataset(root=self.root, phase=self.phase, batch_size=self.batch_size)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.my_collate)

    def name(self):
        return 'SymDataLoader'

    def my_collate(self, batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
