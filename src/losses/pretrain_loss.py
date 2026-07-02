from math import pi
import torch.nn as nn
import torch
from ..models.rotation_utils import *


class PretrainLoss(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

    def forward(self, outputs_model, inputs_data, mode='train'):
        rot_mat_gt = inputs_data['rotation']
        angles_pred = outputs_model['logits']
        angles_gt = matrix_to_euler_angles(rot_mat_gt, 'XYZ')

        loss_x = rotation_loss_class(angles_pred[..., 0], angles_gt[..., 0])
        loss_y = rotation_loss_class(angles_pred[..., 1], angles_gt[..., 1])
        loss_z = rotation_loss_class(angles_pred[..., 2], angles_gt[..., 2])

        loss = loss_x+loss_y+loss_z
        return loss, {'loss_x': loss_x, 'loss_y': loss_y, 'loss_z': loss_z}


def rotation_loss_class(out_rotation_x, angle_x):
    length = out_rotation_x.size(-1)
    label = ((angle_x.view(-1) + pi) / 2 / pi * length)
    label[label < 0] += length
    label[label >= length] -= length
    criterion = nn.CrossEntropyLoss()
    loss_x = criterion(out_rotation_x, label.long())
    return loss_x

