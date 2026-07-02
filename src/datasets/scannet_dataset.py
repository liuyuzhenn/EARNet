from torch.utils.data import Dataset
import torch
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import torch
import os
from torch.utils.data import Dataset


MEAN = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3)).astype(np.float32)
STD = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3)).astype(np.float32)


class ScannetDataset(Dataset):
    def __init__(self, configs, split):
        super().__init__()
        self.args = configs
        self.path = configs['path']
        self.pairs_per_scene = configs['pairs_per_scene']
        self.split = split
        self.height = configs.get('height', 256)
        self.width = configs.get('width', 256)
        # metas: scene_id, img1_id, img2_id
        if split == 'train':
            self.metas_all = self.get_train_metas(self.path)
            self.data_dir = os.path.join(self.path, 'scans')
        else:
            self.metas_all = self.get_test_metas(self.path)
            self.data_dir = os.path.join(self.path, 'scans_test')
        self.inds = []
        self.process_epoch()

    def process_epoch(self):
        self.inds = []
        if self.split == 'train':
            for metas_scene in self.metas_all:
                num = len(metas_scene)
                inds_all = np.arange(num)
                self.inds.append(np.random.choice(inds_all,
                                                  self.pairs_per_scene,
                                                  replace=False))

    def get_test_metas(self, path):
        info = os.path.join(path, 'preprocessed/testdata/test.npz')
        info = np.load(info)
        names = info['name']
        metas = [[n[0], n[2], n[3]] for n in names]
        return metas

    def get_train_metas(self, path):
        min_overlap = self.args['min_overlap']
        metas = []
        scenes = os.listdir(os.path.join(path, 'scans'))
        scenes.sort()
        for s in tqdm(scenes, ncols=80, desc='Loading training split...'):
            info = os.path.join(
                path, 'preprocessed/scannet_indices/scene_data/train/{}.npz'.format(s))
            f = np.load(info)
            name, score = f['name'], f['score']
            mask = score >= min_overlap
            name = name[mask][:, 2:]
            metas.append([[s, n[0], n[1]] for n in name])

        return metas

    def read_img(self, filename):
        img = Image.open(filename)
        img = img.resize([self.width, self.height])
        img = np.array(img, dtype=np.float32)/255.
        img = (img-MEAN)/STD
        # scale 0~255 to 0~1
        np_img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        return np_img

    def __len__(self):
        if self.split == 'train':
            return len(self.metas_all)*self.pairs_per_scene
        else:
            return len(self.metas_all)

    def __getitem__(self, index):
        if self.split == 'train':
            scene_idx = index//self.pairs_per_scene
            pair_idx = self.inds[scene_idx][index % self.pairs_per_scene]
            scene, id1, id2 = self.metas_all[scene_idx][pair_idx]
        else:
            scene, id1, id2 = self.metas_all[index]
            scene = 'scene0{}_00'.format(scene)

        img1_path = os.path.join(self.data_dir, scene, 'color', f'{id1}.jpg')
        img2_path = os.path.join(self.data_dir, scene, 'color', f'{id2}.jpg')
        pose1_path = os.path.join(self.data_dir, scene, 'pose', f'{id1}.txt')
        pose2_path = os.path.join(self.data_dir, scene, 'pose', f'{id2}.txt')
        K_path = os.path.join(self.data_dir, scene,
                              'intrinsic', 'intrinsic_color.txt')

        img1 = self.read_img(img1_path)
        img2 = self.read_img(img2_path)
        imgs = torch.stack([img1, img2], dim=0).float()

        pose1 = np.loadtxt(pose1_path).astype(np.float32)
        pose2 = np.loadtxt(pose2_path).astype(np.float32)
        K = self.read_K(K_path)

        rel_trans = np.linalg.inv(pose2)@pose1
        R = rel_trans[:3, :3]

        return {
            'images': imgs,
            'rotation': R,
        }
