"""
Inspect the four feature maps produced by ViTAdapterEncoder.
Prints shape and value statistics (mean, std, min, max) for each scale.

Also checks the ViTPyramid baseline for comparison.

Usage (from repo root):
    python tools/inspect_adapter_features.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from models.vit_adapter import ViTAdapterEncoder
from models.vit_pyramid import ViTPyramid


def stats(name: str, t: torch.Tensor):
    t = t.float()
    print(f"  {name}: shape={tuple(t.shape)}  "
          f"mean={t.mean():.4f}  std={t.std():.4f}  "
          f"min={t.min():.4f}  max={t.max():.4f}")


@torch.no_grad()
def inspect_adapter(img_size=(640, 640), backbone="vit_base_patch14_reg4_dinov2"):
    print(f"\n=== ViTAdapterEncoder ({backbone}, img_size={img_size}) ===")

    enc = ViTAdapterEncoder(
        backbone_name=backbone,
        img_size=img_size,
        inplanes=64,
        num_heads=8,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        with_cffn=True,
        cffn_ratio=0.25,
        add_vit_feature=True,
        use_extra_extractor=True,
    ).eval()

    H, W = img_size
    x = torch.randn(1, 3, H, W) * 0.5 + 0.5  # simulate [0,1] normalised input

    features = enc(x)
    strides = [4, 8, 16, 32]
    for stride, f in zip(strides, features):
        stats(f"F{stride}", f)
        expected_h, expected_w = H // stride, W // stride
        actual_h, actual_w = f.shape[-2], f.shape[-1]
        if (actual_h, actual_w) != (expected_h, expected_w):
            print(f"    *** STRIDE MISMATCH: expected ({expected_h},{expected_w}), "
                  f"got ({actual_h},{actual_w}) ***")

    print(f"\n  feature_channels expected by pixel_decoder: "
          f"{[enc.backbone.embed_dim] * 4}")


@torch.no_grad()
def inspect_baseline(img_size=(640, 640), backbone="vit_base_patch14_reg4_dinov2",
                     patch_size=14):
    print(f"\n=== ViTPyramid ({backbone}, patch_size={patch_size}, img_size={img_size}) ===")

    enc = ViTPyramid(
        backbone_name=backbone,
        img_size=img_size,
        patch_size=patch_size,
    ).eval()

    H, W = img_size
    x = torch.randn(1, 3, H, W) * 0.5 + 0.5

    features = enc(x)
    strides = [4, 8, 16, 32]
    for stride, f in zip(strides, features):
        stats(f"F{stride}", f)
        expected_h, expected_w = H // stride, W // stride
        actual_h, actual_w = f.shape[-2], f.shape[-1]
        if (actual_h, actual_w) != (expected_h, expected_w):
            print(f"    *** STRIDE MISMATCH: expected ({expected_h},{expected_w}), "
                  f"got ({actual_h},{actual_w}) ***")


if __name__ == "__main__":
    inspect_adapter()
    inspect_baseline()
