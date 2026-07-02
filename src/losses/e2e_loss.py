import torch.nn as nn
import torch.nn.functional as F
import torch
from ..models.rotation_utils import align_rotation
from .exceptions import NoGradientError


class E2eLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs['loss_configs']
        self.type = self.configs['type']
        self.align = self.configs['align']

    def forward(self, outputs_model, inputs_data, mode='train'):
        R = outputs_model['rotations']  # B, 3, 3
        R_gt = inputs_data['rotations']  # B, 3, 3

        if R is None:
            raise NoGradientError

        if mode != 'train':
            R = align_rotation(R, R_gt)
        else:
            if self.align:
                R = align_rotation(R, R_gt)

        if self.type == 'L1':
            diff = R.flatten(-2)-R_gt.flatten(-2)
            loss = torch.norm(diff, p=1, dim=-1).mean()
        elif self.type == 'L2':
            diff = R.flatten(-2)-R_gt.flatten(-2)
            loss = torch.norm(diff, p=2, dim=-1).mean()
        elif self.type == 'geodesic':
            R = R.reshape(-1, 3, 3)
            R_gt = R_gt.reshape(-1, 3, 3)
            R_rel = R@R_gt.transpose(-1, -2)
            tr = torch.stack([torch.trace(r) for r in R_rel])
            loss = -tr.mean()
        else:
            raise NotImplementedError
        return loss
