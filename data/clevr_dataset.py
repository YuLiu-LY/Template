import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

import torch
import random
from PIL import Image
from glob import glob
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from data.datasets import register


class CLEVRDataset(Dataset):
    def __init__(
        self,
        files,
        max_n_objects,
        split:str,
        use_rescale=True,
    ):
        super().__init__()
        self.max_n_objects = max_n_objects
        self.split = split

        self.transform_mask = transforms.Compose(
            [
                # transforms.Resize(128, Image.NEAREST),
                transforms.ToTensor(),
            ]
        )
        T = [
                transforms.ToTensor(),
            ]
        if use_rescale:
            T.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        self.transform_img = transforms.Compose(T)
        
        self.files = files
        
    def __getitem__(self, index: int):
        scene_path = self.files[index]
        with open(scene_path) as f:
                scene_file = json.load(f)
        img_path, mask_paths = self.file_2_paths(scene_file)
        img = Image.open(img_path)
        img = img.convert("RGB")
        image = self.transform_img(img)

        masks = []
        for mask_path in mask_paths:
            masks.append(self.transform_mask(Image.open(mask_path))[None, ...])
        masks = torch.cat(masks, dim=0)
        masks = masks.argmax(dim=0)

        color = torch.Tensor(scene_file['color']).long()[:self.max_n_objects+1]
        shape = torch.Tensor(scene_file['shape']).long()[:self.max_n_objects+1]
        size = torch.Tensor(scene_file['size']).long()[:self.max_n_objects+1]

        return {
            'image': image,
            'mask': masks,
            'color': color,
            'shape': shape,
            'size': size,
        }


    def __len__(self):
        return len(self.files)


    def file_2_paths(self, scene_file):
        # the img_file format: 'num_obj_in_secne' + img_path
        img_path = scene_file['image_path']
        mask_paths = [img_path.replace('images', 'masks').replace('img', 'mask').replace('.png', '_'+str(i)+'.png') for i in range(self.max_n_objects+1)]

        return img_path, mask_paths
              

@register('clevr')
class CLEVRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.max_n_objects = args.max_n_objects

        self.train_dataset, self.val_dataset, self.test_dataset = self.load(args)

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
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def load(self, args):

        if not os.path.exists(args.data_root):
            raise Exception("Data folder does not exist.")

        # Create splits if needed, if you want to create new splits but you have done this before, please change 
        # create_splits into True
        modes = ['train', 'val', 'test']
        # create_splits = True
        create_splits = False
        for m in modes:
            if not os.path.exists(f'{args.data_root}/{m}_images.txt'):
                create_splits = True
                break
        if create_splits:
            print("Creating new train/val/test splits...")
            # Randomly split into train/val/test with fixed seed
            all_scenes_raw = sorted(glob(f'{args.data_root}/scenes/*.json'))
            all_scenes = []
            for scene_path in all_scenes_raw:
                with open(scene_path) as f:
                    scene_file = json.load(f)
                shape = scene_file['shape']
                if shape[self.max_n_objects+1] == 0: 
                    all_scenes.append(scene_path)
            random.seed(0)
            random.shuffle(all_scenes)
            num_eval_scenes = len(all_scenes) // 10
            train_scenes = all_scenes[2*num_eval_scenes:]
            val_scenes = all_scenes[:num_eval_scenes]
            test_scenes = all_scenes[num_eval_scenes:2*num_eval_scenes]
            modes = ['train', 'val', 'test']
            mode_scenes = [train_scenes, val_scenes, test_scenes]
            for mode, mscs in zip(modes, mode_scenes):
                img_paths = mscs
                with open(f'{args.data_root}/{mode}_images.txt', 'w') as f:
                    for item in sorted(img_paths):
                        f.write("%s\n" % item)
            # Sanity checks
            assert len(train_scenes + val_scenes + test_scenes) == len(all_scenes)
            assert not list(set(train_scenes).intersection(val_scenes))
            assert not list(set(train_scenes).intersection(test_scenes))
            assert not list(set(val_scenes).intersection(test_scenes))
            print("Created new train/val/test splits!")

        # Read splits
        with open(f'{args.data_root}/train_images.txt') as f:
            train_images = f.readlines()
            train_images = [x.strip() for x in train_images]
        with open(f'{args.data_root}/val_images.txt') as f:
            val_files = f.readlines()
            val_images = []
            for x in val_files:
                val_images.append(x.strip())
        with open(f'{args.data_root}/test_images.txt') as f:
            test_images = f.readlines()
            test_images = [x.strip() for x in test_images]
        print(f"{len(train_images)} train images")
        print(f"{len(val_images)} val images")
        print(f"{len(test_images)} test images")

        # Datasets
        trainset = CLEVRDataset(train_images, self.max_n_objects, 'train', args.use_rescale)
        valset = CLEVRDataset(val_images, self.max_n_objects, 'val', args.use_rescale)
        testset = CLEVRDataset(test_images, self.max_n_objects, 'test', args.use_rescale)

        return trainset, valset, testset




def crop(x):
    # input: [240, 320, 3]
    # output: [192, 192, 3]
    return x[29:221, 64:256, :]

def process_data(data_root):
    resolution = (128, 128)
    crop_size = [192, 192]
    max_objs = 6
    
    img_T = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(resolution),
    ])
    mask_T = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(resolution),
    ])
    if not os.path.exists(f'{data_root}/images'):
        os.makedirs(f'{data_root}/images')
    if not os.path.exists(f'{data_root}/masks'):
        os.makedirs(f'{data_root}/masks')
    if not os.path.exists(f'{data_root}/scenes'):
        os.makedirs(f'{data_root}/scenes')

    from third_party.multi_object_datasets.clevr_with_masks import dataset as Dataset
    import tensorflow as tf

    tf_records_path = f'{data_root}/clevr_with_masks_train.tfrecords'
    batch_size = 4096
    tf.compat.v1.disable_eager_execution()
    dataset = Dataset(tf_records_path)
    batched_dataset = dataset.batch(batch_size)  # optional batching
    iterator = batched_dataset.make_one_shot_iterator()
    data = iterator.get_next()

    with tf.compat.v1.Session() as sess:
        num = 0
        for _ in tqdm(range(25)):
            batch = sess.run(data)
            image_batch = batch.pop('image')
            mask_batch = batch.pop('mask')
            keys = batch.keys()
            num_image = len(image_batch)
            for i in range(num_image):
                shape = batch['shape'][i]
                if shape[max_objs+1].sum() == 0: # num_obj < max_obj
                    # save image
                    img = image_batch[i]
                    img = img_T(Image.fromarray(crop(img), 'RGB'))
                    image_path = f'{data_root}/images/img_{str(num).zfill(6)}.png'
                    img.save(image_path)
                    # save mask
                    masks = mask_batch[i]
                    for k in range(max_objs+1):
                        mask = masks[k].squeeze()
                        mask = mask_T(Image.fromarray(mask, mode='L'))
                        mask_path = f'{data_root}/masks/mask_{str(num).zfill(6)}_{k}.png'
                        mask.save(mask_path)
                    # save other properties
                    batch.pop
                    dict = {}
                    dict['image_path'] = image_path
                    for k in keys:
                        dict[k] = batch[k][i].tolist()
                    scene_path = f'{data_root}/scenes/scene_{str(num).zfill(6)}.json'
                    with open(scene_path, 'w') as f:
                        f.write(json.dumps(dict, indent=4, ensure_ascii=False))
                    num += 1
            if num_image < batch_size:
                print(f'processed {num} images')
                break
            
'''process data for train. you only need to run it once'''
# DATA_ROOT = '/scratch/generalvision/CLEVR'
# process_data()


'''test'''
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.data_root = '/scratch/generalvision/CLEVR'
    args.use_rescale = False
    args.batch_size = 20
    args.num_workers = 4
    args.max_n_objects = 3

    datamodule = CLEVRDataModule(args)
    dl = datamodule.val_dataloader()
    it = iter(dl)
    batch = next(it)
    batch_img, batch_masks = batch['image'], batch['mask']
    print(batch_img.shape, batch_masks.shape)
    # for mask in batch_masks:
    #     print(mask.unique())
    # print(batch_masks[0, 0, 16:48, 16:32])
    dl = datamodule.train_dataloader()
    print(len(dl))
