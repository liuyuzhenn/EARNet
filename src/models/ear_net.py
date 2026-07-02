import time
from math import pi

import torch
import torch.nn as nn

from src.runners.utils import to_device

from .cara import CARA
from .encoders import EfficientNetV2Encoder, ResNetEncoder
from .fusion import *
from .modules import MLP
from .rotation_utils import *

eps = 1e-4


class EarNet(nn.Module):
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
        if name_fusion == "FeatureFusion":
            self.fusion = FeatureFusion(fusion_configs)
            dim_embedding = fusion_configs["dim_embedding_visual"]

        elif name_fusion == "CVFusion":
            self.fusion = CVFusion(fusion_configs)
            dim_embedding = fusion_configs["dim_embedding_cv"]
        elif name_fusion == "HybridFusion":
            self.fusion = HybridFusion(fusion_configs)
            dim_embedding = (
                fusion_configs["dim_embedding_visual"]
                + fusion_configs["dim_embedding_cv"]
            )
        else:
            raise NotImplementedError

        self.rotation_mode = decoder_configs["rotation_mode"]

        ############################
        # rotation decoding branch #
        ############################
        if self.rotation_mode == "6d":
            self.out_dim = 6
        elif self.rotation_mode == "quaternion":
            self.out_dim = 4
        elif self.rotation_mode == "axis_angle":
            self.out_dim = 3
        elif self.rotation_mode == "euler":
            self.out_dim = 3
        elif self.rotation_mode == "distribution":
            self.out_dim = model_configs["decoder"]["out_dim"]
        else:
            raise NotImplementedError

        channel_list = decoder_configs["rot_dec_channels"]
        channel_list = [dim_embedding] + channel_list + [self.out_dim]
        self.rotation_decoder = MLP(channel_list, True)
        if self.rotation_mode == "distribution":
            self.decoder_y = MLP(channel_list, True)
            self.decoder_z = MLP(channel_list, True)

        ckpt = model_configs.get("pretrained", "")
        if ckpt != "":
            print("Load weight from: {}".format(ckpt))
            ckpt = torch.load(ckpt, map_location="cpu")
            self.load_state_dict(ckpt["model_state_dict"])

        ##########################
        # weight decoding branch #
        ##########################

        weight_dec_channels = decoder_configs.get("weight_dec_channels", None)
        if weight_dec_channels is not None:
            weight_dec_channels = (
                [self.fusion.dim_embedding] + weight_dec_channels + [1]
            )
            self.weight_decoder = MLP(weight_dec_channels, True)
        else:
            self.weight_decoder = None

        self.solver_cfg = model_configs["cara"]
        self.cara = CARA(
            self.solver_cfg["num_iters"],
            self.solver_cfg["cost_fun"],
            sigma=self.solver_cfg.get("sigma", 1),
            threshold=self.solver_cfg.get("threshold", -1),
        )
        self.use_CAI = self.solver_cfg.get("use_CAI", False)
        self.large_graph_inference = self.solver_cfg.get("large_graph_inference", False)
        self.k = model_configs.get("k", -1)
        codes = torch.arange(self.out_dim, dtype=torch.float32) / self.out_dim
        codes = (codes * 2 * pi - pi).reshape(1, self.out_dim, 1)
        self.register_buffer("codes", codes)

    def get_rotation(self, embeddings):
        logits = self.rotation_decoder(embeddings)
        if self.rotation_mode == "6d":
            rotations = rotation_6d_to_matrix(logits)
        elif self.rotation_mode == "quaternion":
            rotations = quaternion_to_matrix(logits)
        elif self.rotation_mode == "axis_angle":
            rotations = axis_angle_to_matrix(logits)
        elif self.rotation_mode == "euler":
            rotations = euler_angles_to_matrix(logits, "XYZ")
        elif self.rotation_mode == "distribution":
            logits_y = self.decoder_y(embeddings)
            logits_z = self.decoder_z(embeddings)
            logits = torch.stack((logits, logits_y, logits_z), dim=-1)  # B,360,3
            logits_softmax = F.softmax(logits, dim=1)
            # B,3
            rotations = (logits_softmax * self.codes).sum(dim=1, keepdims=False)
            rotations = euler_angles_to_matrix(rotations, "XYZ")

        else:
            raise NotImplementedError
        return rotations

    def forward(self, inputs_data, mode="train"):
        x = inputs_data["images"]  # B, N, 3, H, W
        device = x.device
        bs, n, c, h, w = x.shape

        if not self.large_graph_inference:
            ##########################
            # image feature encoding #
            ##########################
            fmaps = [self.encoder(x[:, i]) for i in range(n)]

            #################################
            # pairwise image feature fusion #
            #################################
            pairwise_embeddings = []
            inds = []
            for i in range(n - 1):
                for j in range(i + 1, n):
                    pairwise_feature = self.fusion(fmaps[i], fmaps[j])
                    pairwise_embeddings.append(pairwise_feature)
                    inds.append([i, j])
            inds = torch.tensor(inds, dtype=torch.int64, device=device)
            ############################
            # rotation deocding branch #
            ############################

            # num_pairs*bs,C,H,W
            pairwise_embeddings = torch.cat(pairwise_embeddings, dim=0)
            rotations = self.get_rotation(pairwise_embeddings)
            rotations = rotations.reshape(-1, bs, 3, 3).transpose(1, 0)

            ##############################
            # confidence deocding branch #
            ##############################
            if self.weight_decoder is not None:
                weights = self.weight_decoder(pairwise_embeddings) / 10
                weights = torch.sigmoid(weights)
                weights = weights.reshape(-1, bs).transpose(1, 0)
                # avoid numerical issue
                weights = weights + eps
            else:
                weights = None

            ##############################################
            # CARA (Confidence-Aware Rotation Aevraging) #
            ##############################################
            abs_rotations, num_iters = self.cara.solve(
                rotations, inds, 0, weights, use_CAI=self.use_CAI
            )

            return {
                "rotations": abs_rotations,
                "num_iters": num_iters,
                "weights": weights,
                "rel_rotations": rotations,
                "inds": inds,
            }
        else:
            assert bs == 1, "batch size could only be 1 in large graph inference"
            assert self.k > 0
            ##########################
            # image feature encoding #
            ##########################
            fmaps = [to_device(self.encoder(x[:, i]), "cpu") for i in range(n)]
            inds = []
            rotations = []
            weights = []
            for i in range(n - 1):
                end = min(i + 1 + self.k, n)
                for j in range(i + 1, end):
                    inds.append([i, j])
                    #################################
                    # pairwise image feature fusion #
                    #################################
                    pairwise_feature = self.fusion(
                        to_device(fmaps[i], "cuda"), to_device(fmaps[j], "cuda")
                    )

                    ############################
                    # rotation deocding branch #
                    ############################
                    r = self.get_rotation(pairwise_feature)
                    rotations.append(r[0].cpu())

                    ##############################
                    # confidence deocding branch #
                    ##############################
                    if self.weight_decoder is not None:
                        w = self.weight_decoder(pairwise_feature) / 10
                        w = torch.sigmoid(w) + eps
                        weights.append(w.cpu())

            inds = torch.tensor(inds, dtype=torch.int64, device=device)
            rotations = torch.concat(rotations, dim=0)
            rotations = rotations.reshape(1, -1, 3, 3)

            if self.weight_decoder is None:
                weights = None
            else:
                weights = torch.tensor(weights).unsqueeze(0)

            return {"rel_rotations": rotations, "weights": weights, "inds": inds}

