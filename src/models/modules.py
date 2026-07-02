import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, channel_list, disable_final_activation=False):
        super(MLP, self).__init__()

        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(
                nn.Linear(channel_list[layer_index], channel_list[layer_index + 1])
            )
            layer_list.append(nn.LeakyReLU(inplace=True))

        if disable_final_activation:
            layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.net(x)
