# ---------------------------------------------------------------
# Adapted from czczup/ViT-Adapter (adapter_modules.py + vit_adapter.py)
# Original: Copyright (c) Shanghai AI Lab. All rights reserved.
#
# Modifications for DINOv2 (patch_size=14, prefix tokens) and
# removal of mmcv/MSDeformAttn dependency (replaced with MHA).
# ---------------------------------------------------------------

import math
from functools import partial
from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_


# ---------------------------------------------------------------------------
# Spatial Prior Module
# ---------------------------------------------------------------------------

class SpatialPriorModule(nn.Module):
    """
    Small CNN that extracts multi-scale features from the raw input image.
    Outputs:
      c1: (B, embed_dim, H/4,  W/4)  — kept as spatial tensor
      c2: (B, H/8*W/8,   embed_dim)  — flattened sequence
      c3: (B, H/16*W/16, embed_dim)  — flattened sequence
      c4: (B, H/32*W/32, embed_dim)  — flattened sequence
    """

    def __init__(self, inplanes: int = 64, embed_dim: int = 384):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(4 * inplanes),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        c1 = self.stem(x)    # (B, inplanes,   H/4,  W/4)
        c2 = self.conv2(c1)  # (B, 2*inplanes, H/8,  W/8)
        c3 = self.conv3(c2)  # (B, 4*inplanes, H/16, W/16)
        c4 = self.conv4(c3)  # (B, 4*inplanes, H/32, W/32)

        c1 = self.fc1(c1)    # (B, embed_dim, H/4,  W/4) — spatial
        c2 = self.fc2(c2).flatten(2).transpose(1, 2)  # (B, H/8*W/8,   D)
        c3 = self.fc3(c3).flatten(2).transpose(1, 2)  # (B, H/16*W/16, D)
        c4 = self.fc4(c4).flatten(2).transpose(1, 2)  # (B, H/32*W/32, D)
        return c1, c2, c3, c4


# ---------------------------------------------------------------------------
# ConvFFN + DWConv  (used inside Extractor)
# ---------------------------------------------------------------------------

class DWConv(nn.Module):
    """
    Depthwise conv applied separately to each SPM scale within the
    concatenated sequence c = cat[c2, c3, c4].

    spm_sizes: [(h8,w8), (h16,w16), (h32,w32)] — explicit spatial sizes,
    replacing the original's (H*2, H, H//2) which assumed patch_size=16.
    """

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                bias=True, groups=dim)

    def forward(self, x: torch.Tensor, spm_sizes: list) -> torch.Tensor:
        B, N, C = x.shape
        (h2, w2), (h3, w3), (h4, w4) = spm_sizes
        n2, n3 = h2 * w2, h3 * w3

        x1 = x[:, :n2].transpose(1, 2).view(B, C, h2, w2)
        x2 = x[:, n2:n2 + n3].transpose(1, 2).view(B, C, h3, w3)
        x3 = x[:, n2 + n3:].transpose(1, 2).view(B, C, h4, w4)

        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        return torch.cat([x1, x2, x3], dim=1)


class ConvFFN(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, spm_sizes: list) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, spm_sizes)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ---------------------------------------------------------------------------
# Injector  (SPM → ViT patch tokens)
# ---------------------------------------------------------------------------

class Injector(nn.Module):
    """
    Injects SPM multi-scale context into ViT patch tokens via cross-attention.
    query = patch tokens  (B, N_patch, D)
    feat  = c = cat[c2,c3,c4]  (B, N_spm, D)

    Replaces original MSDeformAttn with standard nn.MultiheadAttention.
    """

    def __init__(self, dim: int, num_heads: int = 6, init_values: float = 0.):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.feat_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, query: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.feat_norm(feat),
            self.feat_norm(feat),
        )
        return query + self.gamma * attn_out


# ---------------------------------------------------------------------------
# Extractor  (ViT patch tokens → SPM features)
# ---------------------------------------------------------------------------

class Extractor(nn.Module):
    """
    Extracts updated context from ViT patch tokens back into SPM features
    via cross-attention.
    query = c = cat[c2,c3,c4]  (B, N_spm, D)
    feat  = patch tokens        (B, N_patch, D)

    Replaces original MSDeformAttn with standard nn.MultiheadAttention.
    """

    def __init__(self, dim: int, num_heads: int = 6, with_cffn: bool = True,
                 cffn_ratio: float = 0.25, drop: float = 0., drop_path: float = 0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.with_cffn = with_cffn
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim,
                               hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query: torch.Tensor, feat: torch.Tensor,
                spm_sizes: list) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(query),
            self.feat_norm(feat),
            self.feat_norm(feat),
        )
        query = query + attn_out
        if self.with_cffn:
            query = query + self.drop_path(self.ffn(self.ffn_norm(query), spm_sizes))
        return query


# ---------------------------------------------------------------------------
# InteractionBlock
# ---------------------------------------------------------------------------

class InteractionBlock(nn.Module):
    """
    One interaction stage: inject SPM → run ViT blocks → extract back to SPM.

    forward() receives the full token sequence (prefix + patch) so that
    ViT blocks can attend across all tokens (CLS, registers, patches).
    The injector/extractor operate on patch tokens only.
    """

    def __init__(self, dim: int, num_heads: int = 6, init_values: float = 0.,
                 drop: float = 0., drop_path: float = 0., with_cffn: bool = True,
                 cffn_ratio: float = 0.25, extra_extractor: bool = False,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.injector = Injector(dim=dim, num_heads=num_heads, init_values=init_values)
        self.extractor = Extractor(dim=dim, num_heads=num_heads, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
                                   norm_layer=norm_layer)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, with_cffn=with_cffn,
                          cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path,
                          norm_layer=norm_layer)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, tokens: torch.Tensor, c: torch.Tensor,
                blocks: list, spm_sizes: list, num_prefix: int):
        """
        tokens:     (B, num_prefix + N_patch, D)  full ViT token sequence
        c:          (B, N_spm, D)                 concatenated SPM features
        blocks:     list of ViT blocks to run for this stage
        spm_sizes:  [(h8,w8), (h16,w16), (h32,w32)]
        num_prefix: number of prefix tokens (CLS + registers)

        Returns:
          tokens:  updated full token sequence
          c:       updated SPM features
        """
        prefix = tokens[:, :num_prefix]
        patch = tokens[:, num_prefix:]

        # Inject SPM context into patch tokens
        patch = self.injector(query=patch, feat=c)

        # Run ViT blocks with full token sequence (prefix attends to patch and vice versa)
        tokens = torch.cat([prefix, patch], dim=1)
        for blk in blocks:
            tokens = blk(tokens)

        # Extract updated context from post-block patch tokens into SPM
        patch = tokens[:, num_prefix:]
        c = self.extractor(query=c, feat=patch, spm_sizes=spm_sizes)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, feat=patch, spm_sizes=spm_sizes)

        return tokens, c


# ---------------------------------------------------------------------------
# ViTAdapterEncoder
# ---------------------------------------------------------------------------

class ViTAdapterEncoder(nn.Module):
    """
    DINOv2 ViT backbone augmented with ViT-Adapter:
      - SpatialPriorModule (SPM) extracts multi-scale CNN features
      - 4 InteractionBlocks inject/extract between SPM and ViT tokens
      - Outputs 4 feature maps at strides [4, 8, 16, 32]

    Exposes the attribute tree required by EoMT's LightningModule LLRD:
      self.backbone.embed_dim
      self.backbone.blocks       (nn.Sequential of timm ViT blocks)
      self.backbone.num_prefix_tokens
      self.backbone.patch_embed.grid_size
    """

    def __init__(
        self,
        backbone_name: str = "vit_large_patch14_reg4_dinov2",
        img_size: tuple = (640, 640),
        inplanes: int = 64,
        num_heads: int = 8,
        init_values: float = 0.,
        interaction_indexes: Optional[list] = None,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        add_vit_feature: bool = True,
        use_extra_extractor: bool = True,
        drop_path_rate: float = 0.,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            num_classes=0,
        )
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = state_dict.get("model", state_dict)
            self.backbone.load_state_dict(state_dict, strict=False)

        embed_dim = self.backbone.embed_dim
        self.add_vit_feature = add_vit_feature
        self.interaction_indexes = interaction_indexes or [[0, 5], [6, 11], [12, 17], [18, 23]]

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=inplanes, embed_dim=embed_dim)

        self.interactions = nn.Sequential(*[
            InteractionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                init_values=init_values,
                drop_path=drop_path_rate,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                extra_extractor=(i == len(self.interaction_indexes) - 1 and use_extra_extractor),
            )
            for i in range(len(self.interaction_indexes))
        ])

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(embed_dim)
        self.norm2 = nn.BatchNorm2d(embed_dim)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.norm4 = nn.BatchNorm2d(embed_dim)

        self._init_adapter_weights()

    def _init_adapter_weights(self):
        """Initialise adapter components (SPM, interactions, up-conv).
        The backbone keeps its pretrained weights.
        """
        for m in [self.spm, self.interactions, self.up]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
                    nn.init.constant_(module.bias, 0)
                    nn.init.constant_(module.weight, 1.0)
                elif isinstance(module, nn.MultiheadAttention):
                    trunc_normal_(module.in_proj_weight, std=0.02)
                    if module.in_proj_bias is not None:
                        nn.init.constant_(module.in_proj_bias, 0)
                elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                    fan_out = (module.kernel_size[0] * module.kernel_size[1]
                               * module.out_channels // module.groups)
                    module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                    if module.bias is not None:
                        module.bias.data.zero_()
        normal_(self.level_embed)

    def forward(self, x: torch.Tensor) -> list:
        """
        x: (B, 3, H, W) — ImageNet-normalised, float32

        Returns [f1, f2, f3, f4] feature maps at strides [4, 8, 16, 32],
        each with embed_dim channels.
        """
        B, _, H, W = x.shape
        spm_sizes = [(H // 8, W // 8), (H // 16, W // 16), (H // 32, W // 32)]
        patch_size = self.backbone.patch_embed.patch_size[0]
        H_patch = H // patch_size
        W_patch = W // patch_size

        # ---- SPM ----
        c1, c2, c3, c4 = self.spm(x)
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        c = torch.cat([c2, c3, c4], dim=1)  # (B, N2+N3+N4, D)

        # ---- Patch embedding + positional embedding ----
        tokens = self.backbone.patch_embed(x)        # (B, N_patch, D)
        tokens = self.backbone._pos_embed(tokens)    # (B, num_prefix + N_patch, D)
        num_prefix = self.backbone.num_prefix_tokens

        # ---- 4 interaction stages ----
        outs = []
        for i, interaction in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            blocks = list(self.backbone.blocks[indexes[0]:indexes[-1] + 1])
            tokens, c = interaction(tokens, c, blocks, spm_sizes, num_prefix)
            patch = tokens[:, num_prefix:]
            outs.append(patch.transpose(1, 2).view(B, -1, H_patch, W_patch))

        # ---- Reconstruct SPM spatial maps from updated c ----
        n2 = spm_sizes[0][0] * spm_sizes[0][1]
        n3 = spm_sizes[1][0] * spm_sizes[1][1]
        c2_out = c[:, :n2].transpose(1, 2).view(B, -1, *spm_sizes[0])
        c3_out = c[:, n2:n2 + n3].transpose(1, 2).view(B, -1, *spm_sizes[1])
        c4_out = c[:, n2 + n3:].transpose(1, 2).view(B, -1, *spm_sizes[2])
        c1_out = self.up(c2_out) + c1  # stride 4: (B, D, H/4, W/4)

        # ---- Add ViT intermediate features (patch_size-agnostic interpolation) ----
        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            c1_out = c1_out + F.interpolate(x1, size=(H // 4,  W // 4),  mode="bilinear", align_corners=False)
            c2_out = c2_out + F.interpolate(x2, size=(H // 8,  W // 8),  mode="bilinear", align_corners=False)
            c3_out = c3_out + F.interpolate(x3, size=(H // 16, W // 16), mode="bilinear", align_corners=False)
            c4_out = c4_out + F.interpolate(x4, size=(H // 32, W // 32), mode="bilinear", align_corners=False)

        return [
            self.norm1(c1_out),  # stride 4
            self.norm2(c2_out),  # stride 8
            self.norm3(c3_out),  # stride 16
            self.norm4(c4_out),  # stride 32
        ]
