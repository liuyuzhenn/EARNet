import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

from . import preact_resnet

# stride: 4, 8, 16, 32
resnet_num_channels = {
    "PreActResNet18": [64, 128, 256, 512],
    "PreActResNet34": [64, 128, 256, 512],
    "PreActResNet50": [256, 512, 1024, 2048],
    "PreActResNet101": [256, 512, 1024, 2048],
    "PreActResNet152": [256, 512, 1024, 2048],
}


# stride: 4, 8, 16, 32
efficientnet_num_channels = {
    "EfficientnetV2S": [48, 64, 128, 256],
    "EfficientnetV2M": [48, 80, 160, 304],
    "EfficientnetV2L": [64, 96, 192, 384],
}


def _name_to_class(name):
    return "".join(n.capitalize() for n in name.split("_"))


class ResNetEncoder(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        name = configs["name"]
        model = getattr(preact_resnet, name)()
        self.feature_type = configs["feature_type"]
        self.feature_type.sort()

        idx = [int(t[-1]) for t in self.feature_type]
        self.output_dims = {
            t: resnet_num_channels[name][i - 1] for i, t in zip(idx, self.feature_type)
        }

        self.body = create_feature_extractor(
            model, return_nodes={l: l for l in self.feature_type}
        )

    def forward(self, x):
        "x: b,3,h,w"
        feat = self.body(x)
        return feat


class EfficientNetV2Encoder(nn.Module):
    feature_names = {
        "layer0": "features.0",  # 2
        "layer1": "features.2",  # 4
        "layer2": "features.3",  # 8
        "layer3": "features.4",  # 16
        "layer4": "features.6",  # 32
    }

    def __init__(self, configs) -> None:
        super().__init__()
        name = configs["name"]
        self.feature_type = configs["feature_type"]
        self.feature_type.sort()
        self.name = _name_to_class(name)

        ##########################
        # get efficientnet model #
        ##########################
        model = getattr(models, name)(weights="IMAGENET1K_V1")

        ###############################
        # only keep feature extractor #
        ###############################
        self.body = create_feature_extractor(
            model, return_nodes={self.feature_names[l]: l for l in self.feature_type}
        )

        idx = [int(t[-1]) for t in self.feature_type]
        self.output_dims = {
            t: efficientnet_num_channels[self.name][i - 1]
            for i, t in zip(idx, self.feature_type)
        }

    def forward(self, x):
        "x: b,3,h,w"
        feat = self.body(x)
        return feat

