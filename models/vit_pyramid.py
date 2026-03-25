import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from models.scale_block import ScaleBlock


class ViTPyramid(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size=16,
        backbone_name="vit_large_patch14_reg4_dinov2",
        scales=[4, 8, 16, 32],
    ):
        super().__init__()
        self.scales = scales

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,
        )

        pixel_mean = torch.tensor(self.backbone.default_cfg["mean"]).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(self.backbone.default_cfg["std"]).reshape(1, -1, 1, 1)

        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        patch_size = self.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])

        feature_pyramid = []
        for scale in scales:
            if max_patch_size == scale:
                feature_pyramid.append(nn.Identity())
                continue

            if max_patch_size > scale:
                conv1_layer = nn.ConvTranspose2d
                num_blocks = int(math.log2(max_patch_size / scale))
            else:
                conv1_layer = nn.Conv2d
                num_blocks = int(math.log2(scale / max_patch_size))

            feature_pyramid.append(
                nn.Sequential(
                    *[
                        ScaleBlock(self.backbone.embed_dim, conv1_layer)
                        for _ in range(num_blocks)
                    ],
                )
            )

        self.feature_pyramid = nn.ModuleList(feature_pyramid)

    def forward(self, x):
        x = (x - self.pixel_mean) / self.pixel_std

        x = self.backbone.forward_features(x)

        x = x[:, self.backbone.num_prefix_tokens :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.backbone.patch_embed.grid_size
        )

        return [scale(x) for scale in self.feature_pyramid]
