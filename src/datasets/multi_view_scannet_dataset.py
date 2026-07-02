from torch.utils.data import Dataset
import torch
import time
from PIL import Image
import numpy as np
import torch
import os
from torch.utils.data import Dataset


MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)).astype(np.float32)
STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)).astype(np.float32)


class MultiViewScannetDataset(Dataset):
    def __init__(self, configs, split):
        super().__init__()
        self.args = configs
        self.path = configs['path']
        self.split = split
        self.height = configs.get('height', 256)
        self.width = configs.get('width', 256)
        self.list_path = configs['list']
        self.size_tuple = configs['size_tuple']
        if isinstance(self.size_tuple, list):
            self.size_tuple, self.increment, self.interval, self.max_size = self.size_tuple
            if self.split == 'test':
                self.size_tuple = self.max_size
        else:
            self.increment, self.interval = 0, 1
            self.max_size = self.size_tuple

        if split == 'train':
            self.data_dir = os.path.join(self.path, 'scans')
            self.list_path = os.path.join(self.list_path, 'train')
        elif split == 'test' or  split == 'val':
            self.data_dir = os.path.join(self.path, 'scans_test')
            subdir = configs.get('subdir', None)
            if subdir is None:
                self.list_path = os.path.join(self.list_path, 'test')
            else:
                self.list_path = os.path.join(self.list_path, subdir)
        else:
            raise NotImplementedError
        self.metas = self.build_metas()
        self.epochs = 1

    def build_metas(self):
        metas = []
        list_files = os.listdir(self.list_path)
        for fi in list_files:
            scene = fi.split('.')[0]
            fi = os.path.join(self.list_path, fi)
            with open(fi, 'r') as f:
                lines = f.readlines()
                lines = [l.strip() for l in lines]
            for l in lines:
                inds = [int(i) for i in l.split(' ')]
                metas.append([scene]+inds)
        return metas


    def read_img(self, filename):
        img = Image.open(filename)
        img = img.resize([self.width, self.height])
        img = np.array(img, dtype=np.float32)/255.
        img = (img-MEAN)/STD
        # scale 0~255 to 0~1
        np_img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        return np_img


    def process_epoch(self):
        if self.split == 'train':
            if self.epochs % self.interval == 0:
                self.size_tuple += self.increment
            if self.size_tuple > self.max_size:
                self.size_tuple = self.max_size
        self.epochs += 1


    def __len__(self):
        return len(self.metas)


    def __getitem__(self, index):
        meta = self.metas[index]
        scene = meta[0]
        inds = meta[1:self.size_tuple+1]
        imgs = []
        rotations = []
        for ind in inds:
            img_path = os.path.join(
                self.data_dir, scene, 'color', f'{ind}.jpg')
            pose_path = os.path.join(
                self.data_dir, scene, 'pose', f'{ind}.txt')
            img = self.read_img(img_path)
            imgs.append(img)

            pose = np.loadtxt(pose_path).astype(np.float32)
            R = pose[:3, :3]
            rotations.append(R.T)

        imgs = torch.stack(imgs)  # N,3,H,W
        rotations = torch.from_numpy(np.stack(rotations)).float()  # N,3,3
        rotations = rotations@rotations[:1].transpose(-1, -2)

        return {
            'images': imgs,
            'rotations': rotations,
        }
