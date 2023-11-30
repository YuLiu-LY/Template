from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms


class MyDataset(CIFAR10):
    def __init__(self, cfg, split='train'):
        if split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.15, 0.1, 0.1)], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(31, 2)], p=0.5),
                ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2470, 0.2435, 0.2616]),
            ])
        else:
            transform = transforms.Compose([
                ToTensor(),
                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                        std=[0.2470, 0.2435, 0.2616]),
            ])
        super().__init__(cfg.data_root, train=(split=='train'), download=True, transform=transform)

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return {'image': img, 'label': label}
    

import torch
import numpy as np
from PIL import Image
from glob import glob
from torch.utils.data import Dataset


class CostumDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split:str,
        use_aug=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.split = split

        self.get_files()

        mu = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        self.T1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mu, std=std),
        ])
        if use_aug:
            self.T2 = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.ColorJitter(0.3, 0.15, 0.1, 0.1)], p=0.5),
                transforms.RandomApply([transforms.GaussianBlur(31, 2)], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mu, std=std),
            ])
        else:
            self.T2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mu, std=std),
        ])
        
    def __getitem__(self, index: int):
        out = {}
        path = self.img_files[index]
        if self.split ==  'train':
            img = Image.open(path).convert("RGB")
            img = self.T2(img)
        else:
            img = Image.open(path).convert("RGB")
            img = self.T1(img)
        label = self.labels[index]
        out['label'] = torch.tensor(label).int()
        out['image'] = img
        return out

    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        self.img_files = sorted(glob(f'{self.data_root}/{self.split}/*.png'))
        self.labels = [np.loadtxt(x.replace('img', 'label').replace('.png', '.txt')) for x in self.img_files]