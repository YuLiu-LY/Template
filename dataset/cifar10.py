from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


class MyDataset(CIFAR10):
    def __init__(self, cfg, split='train'):
        super().__init__(cfg.data_root, train=(split=='train'), download=True, transform=ToTensor())

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return {'image': img, 'target': target}