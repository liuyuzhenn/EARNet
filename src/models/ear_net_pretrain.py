import time
from math import pi

import torch
import torch.nn as nn

from .encoders import EfficientNetV2Encoder, ResNetEncoder
from .fusion import CVFusion
from .modules import MLP
from .rotation_utils import *


class EarNetPretrain(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.configs = configs
        model_configs = configs["model_configs"]
        encoder_configs = model_configs["encoder"]
        fusion_configs = model_configs["fusion"]
        decoder_configs = model_configs["decoder"]

        ###################
        # feature encoder #
        ###################
        if fusion_configs["name"] == "FeatureFusion":
            ftype = [fusion_configs["feature_type_visual"]]
        elif fusion_configs["name"] == "CVFusion":
            ftype = [fusion_configs["feature_type_cv"]]
        elif fusion_configs["name"] == "HybridFusion":
            ftype = [
                fusion_configs["feature_type_visual"],
                fusion_configs["feature_type_cv"],
            ]
        else:
            raise NotImplementedError

        encoder_configs["feature_type"] = ftype

        if "resnet" in model_configs["encoder"]["name"].lower():
            self.encoder = ResNetEncoder(encoder_configs)
        elif "efficientnet_v2" in model_configs["encoder"]["name"].lower():
            self.encoder = EfficientNetV2Encoder(encoder_configs)
        else:
            raise NotImplementedError

        ########################
        # cross feature fusion #
        ########################
        fusion_configs["input_dims"] = self.encoder.output_dims
        name_fusion = fusion_configs["name"]
        if name_fusion == "CVFusion":
            self.fusion = CVFusion(fusion_configs)
        else:
            raise NotImplementedError
        dim_embedding = self.fusion.dim_embedding

        self.rotation_mode = decoder_configs["rotation_mode"]

        ############################
        # rotation decoding branch #
        ############################
        if self.rotation_mode == "6d":
            self.dim_out = 6
        elif self.rotation_mode == "quaternion":
            self.dim_out = 4
        elif self.rotation_mode == "axis_angle":
            self.dim_out = 3
        elif self.rotation_mode == "euler":
            self.dim_out = 3
        elif self.rotation_mode == "distribution":
            self.dim_out = model_configs["decoder"]["out_dim"]
        else:
            raise NotImplementedError

        channel_list = decoder_configs["rot_dec_channels"]
        channel_list = [dim_embedding] + channel_list + [self.dim_out]
        self.rotation_decoder = MLP(channel_list, True)
        if self.rotation_mode == "distribution":
            self.decoder_y = MLP(channel_list, True)
            self.decoder_z = MLP(channel_list, True)

        ckpt = model_configs.get("pretrained", "")
        if ckpt != "":
            print("Load weight from: {}".format(ckpt))
            ckpt = torch.load(ckpt, map_location="cpu")
            self.load_state_dict(ckpt["model_state_dict"])

    def forward(self, inputs_data, mode="train"):
        x = inputs_data["images"]  # B, 2, 3, H, W

        torch.cuda.synchronize()
        f1 = self.encoder(x[:, 0])
        f2 = self.encoder(x[:, 1])
        torch.cuda.synchronize()
        pairwise_embedding = self.fusion(f1, f2)
        torch.cuda.synchronize()
        logits = self.rotation_decoder(pairwise_embedding)
        torch.cuda.synchronize()

        if self.rotation_mode == "6d":
            R = rotation_6d_to_matrix(logits)
        elif self.rotation_mode == "quaternion":
            R = quaternion_to_matrix(logits)
        elif self.rotation_mode == "axis_angle":
            R = axis_angle_to_matrix(logits)
        elif self.rotation_mode == "euler":
            R = euler_angles_to_matrix(logits, "XYZ")
        elif self.rotation_mode == "distribution":
            logits_y = self.decoder_y(pairwise_embedding)
            logits_z = self.decoder_z(pairwise_embedding)

            _, rotation_x = torch.topk(logits, 1, dim=-1)
            _, rotation_y = torch.topk(logits_y, 1, dim=-1)
            _, rotation_z = torch.topk(logits_z, 1, dim=-1)

            rotation_x = rotation_x.squeeze(-1) / self.dim_out * 2 * pi - pi
            rotation_y = rotation_y.squeeze(-1) / self.dim_out * 2 * pi - pi
            rotation_z = rotation_z.squeeze(-1) / self.dim_out * 2 * pi - pi

            angles = torch.stack((rotation_x, rotation_y, rotation_z), dim=-1)
            R = euler_angles_to_matrix(angles, "XYZ")
            logits = torch.stack((logits, logits_y, logits_z), dim=-1)  # B,360,3
        else:
            raise NotImplementedError

        return {"rotation": R, "logits": logits}

