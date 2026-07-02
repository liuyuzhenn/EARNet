import torch
import torch.nn as nn
import torch.nn.functional as F

from .preact_resnet import PreActBlock, PreActBottleneck


def compute_correlation_volume_pairwise(fmap1, fmap2, num_levels=1):
    batch, dim, ht, wd = fmap1.shape
    fmap1 = fmap1.view(batch, dim, ht * wd)
    fmap2 = fmap2.view(batch, dim, ht * wd)

    corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
    corr = corr.view(batch, ht, wd, 1, ht, wd)
    corr = corr / torch.sqrt(torch.tensor(dim).float())

    batch2, h1, w1, dim2, h2, w2 = corr.shape
    corr = corr.reshape(batch2 * h1 * w1, dim2, h2, w2)
    corr_pyramid = []
    corr_pyramid.append(corr)
    for i in range(num_levels - 1):
        corr = F.avg_pool2d(corr, 2, stride=2)
        corr_pyramid.append(corr)

    out_pyramid = []
    for i in range(num_levels):
        corr = corr_pyramid[i]
        corr = corr.view(batch2, h1, w1, -1)
        out_pyramid.append(corr)
        out = torch.cat(out_pyramid, dim=-1)
    return out.permute(0, 3, 1, 2).contiguous().float()


class CVFusion(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        block = [PreActBlock, PreActBottleneck][configs["block"]]

        ##############################
        # geometry feature embedding #
        ##############################
        self.feature_type_cv = configs["feature_type_cv"]
        self.cv_input_size = configs.get("cv_input_size", (16, 16))
        self.cv_dim = self.cv_input_size[0] * self.cv_input_size[1]
        self.in_planes = self.cv_dim
        num_blocks = configs["num_blocks_cv"]
        num_channels = configs["num_channels_cv"]
        dim_embedding = configs["dim_embedding_cv"]
        self.dim_embedding = dim_embedding
        layers = []
        for i in range(len(num_channels)):
            layer = self._make_layer(block, num_channels[i], num_blocks[i], stride=2)
            layers.append(layer)
        layers = nn.Sequential(*layers)
        embed_layer = nn.Sequential(
            nn.BatchNorm2d(num_channels[-1] * block.expansion),
            nn.LeakyReLU(),
            nn.Conv2d(
                num_channels[-1] * block.expansion,
                dim_embedding,
                kernel_size=1,
                bias=False,
            ),
        )
        self.geom_fuser = nn.Sequential(layers, embed_layer)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, feat1, feat2):
        feat1 = feat1[self.feature_type_cv]
        feat2 = feat2[self.feature_type_cv]
        # bs, c, s, s
        feat1_pool = F.adaptive_avg_pool2d(feat1, self.cv_input_size)
        feat2_pool = F.adaptive_avg_pool2d(feat2, self.cv_input_size)

        # bs, s^2, s, s
        cv_volume = compute_correlation_volume_pairwise(feat1_pool, feat2_pool)
        geom_feat = self.geom_fuser(cv_volume)
        geom_feat = F.adaptive_avg_pool2d(geom_feat, 1)
        return geom_feat.flatten(1)
