import os
import torch
import pytorch_lightning as pl
from PIL import Image
from pathlib import Path
from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode


class VOCDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        resolution: Tuple[int, int],
        split: str = "train",
        max_class_per_img: int = 6,
        use_rescale: bool = True,
    ):
        super().__init__()
        self.split = split
        self.resolution = resolution
        if self.split == 'train':
            trans = [
                transforms.RandomResizedCrop(resolution, scale=(0.4, 1.0)),   
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
            ]
        else:
            trans = [
                transforms.Resize(resolution),
                transforms.ToTensor(),
            ]
        if use_rescale:
            trans.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.transform = transforms.Compose(trans)
        self.transform_seg = transforms.Compose([
                transforms.Resize(resolution, interpolation=InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])
        
        seg_folder = "SegmentationClass"
        seg_dir = os.path.join(data_root, seg_folder)
        image_dir = os.path.join(data_root, 'JPEGImages')
        splits_dir = os.path.join(data_root, 'ImageSets/Segmentation')
        split_f = os.path.join(splits_dir, self.split + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(seg_dir, x + ".png") for x in file_names]

        self.max_class_per_img = max_class_per_img

        assert all([Path(f).is_file() for f in self.masks]) and all([Path(f).is_file() for f in self.images])

    def __getitem__(self, index: int):
        img = self.transform(Image.open(self.images[index]).convert('RGB'))
        
        mask = Image.open(self.masks[index])
        mask = self.transform_seg(mask)
        mask = (mask != 1) * mask
        mask = (mask * 255).int().view(1, *self.resolution)
        label = mask.unique() 

        num_pad = self.max_class_per_img - label.shape[0]
        if num_pad > 0:
            label = torch.cat((label, torch.ones(num_pad) * 21), dim=0)
        label = label.long()
        # onehot_label = F.one_hot(label.long(), num_classes=22).float()
        return {
            'image': img,
            'mask': mask, 
            # 'mask_file': self.masks[index],
            'label': label
        }   


    def __len__(self) -> int:
        return len(self.images)


class VOCDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.data_root = args.data_root
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_dataset = VOCDataset(args.data_root, args.resolution, 'train', 6, use_rescale=args.use_rescale)
        self.val_dataset = VOCDataset(args.data_root, args.resolution, 'val', 6, use_rescale=args.use_rescale)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

'''test'''
if __name__ == '__main__':
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_root = '/home/yuliu/Dataset/VOC2012'
    args.use_rescale = True
    args.batch_size = 16
    args.num_workers = 0
    args.resolution = 128, 128

    datamodule = VOCDataModule(args)
    dl = datamodule.val_dataloader()
    batch = next(iter(dl))
    batch_img, batch_masks = batch['image'], batch['mask']
    labels = batch['label']
    print(labels.shape)
    print(labels)