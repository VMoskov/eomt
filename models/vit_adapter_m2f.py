# ---------------------------------------------------------------
# © 2025. Licensed under the MIT License.
#
# ViT-Adapter + HuggingFace Mask2Former for universal segmentation.
# Plugs into EoMT's LightningModule without any changes to existing code.
# ---------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
    Mask2FormerTransformerModule,
)

from models.vit_adapter import ViTAdapterEncoder


class ViTAdapterM2F(nn.Module):
    """
    ViT-Adapter encoder + HuggingFace Mask2Former decoder.

    Satisfies the EoMT LightningModule network interface:
      - forward(x) -> ([mask_logits], [class_logits])
      - masked_attn_enabled = False  (no annealing code paths triggered)
      - num_blocks = 1               (single-pass, no iterative block loss)
      - encoder.backbone.*           (timm ViT attributes for LLRD)
    """

    masked_attn_enabled: bool = False
    num_blocks: int = 1

    def __init__(
        self,
        backbone_name: str = "vit_large_patch14_reg4_dinov2",
        img_size: tuple = (640, 640),
        num_classes: int = 133,
        num_queries: int = 200,
        # ViT-Adapter encoder hyperparams
        inplanes: int = 64,
        num_heads: int = 8,
        init_values: float = 0.,
        interaction_indexes: Optional[list] = None,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        add_vit_feature: bool = True,
        use_extra_extractor: bool = True,
        drop_path_rate: float = 0.,
        backbone_ckpt_path: Optional[str] = None,
        # M2F decoder hyperparams
        feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ---- Encoder: ViT-Adapter ----
        self.encoder = ViTAdapterEncoder(
            backbone_name=backbone_name,
            img_size=img_size,
            inplanes=inplanes,
            num_heads=num_heads,
            init_values=init_values,
            interaction_indexes=interaction_indexes,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            add_vit_feature=add_vit_feature,
            use_extra_extractor=use_extra_extractor,
            drop_path_rate=drop_path_rate,
            ckpt_path=backbone_ckpt_path,
        )

        embed_dim = self.encoder.backbone.embed_dim  # e.g. 1024 for ViT-L

        # ---- M2F decoder config ----
        # feature_strides=[4,8,16,32] matches ViT-Adapter's 4-scale output.
        # common_stride=4 (default) → mask_features at H/4, W/4.
        config = Mask2FormerConfig(
            feature_strides=[4, 8, 16, 32],
            feature_size=feature_size,
            mask_feature_size=feature_size,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            num_queries=num_queries,
            num_labels=num_classes,
            dropout=dropout,
            use_auxiliary_loss=False,
        )

        # ---- Pixel decoder (multi-scale deformable FPN) ----
        self.pixel_decoder = Mask2FormerPixelDecoder(
            config, feature_channels=[embed_dim] * 4
        )

        # ---- Transformer decoder (masked cross-attention queries) ----
        # in_features=feature_size=hidden_dim → input projections are identity
        self.transformer_module = Mask2FormerTransformerModule(
            in_features=feature_size, config=config
        )

        # ---- Class prediction head ----
        self.class_predictor = nn.Linear(hidden_dim, num_classes + 1)

        # ---- ImageNet normalisation buffers (applied in forward) ----
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W) float32 in [0, 1]  — from LightningModule.forward (imgs/255)

        Returns:
          ([mask_logits], [class_logits])
          mask_logits:  (B, num_queries, H/4, W/4)
          class_logits: (B, num_queries, num_classes+1)
        """
        x = x.float()

        # ImageNet normalisation
        x = (x - self.pixel_mean) / self.pixel_std

        # ViT-Adapter encoder runs in fp16 (AMP handles this)
        # → [f1(s4), f2(s8), f3(s16), f4(s32)]
        features = self.encoder(x)

        # HuggingFace pixel + transformer decoder run in fp32 — MSDA overflows in fp16
        with torch.autocast(device_type="cuda", enabled=False):
            features = [f.float() for f in features]
            pix_out = self.pixel_decoder(features)
            trans_out = self.transformer_module(
                multi_scale_features=list(pix_out.multi_scale_features),
                mask_features=pix_out.mask_features,
                output_hidden_states=False,
            )
            class_logits = self.class_predictor(trans_out.last_hidden_state)
            mask_logits = trans_out.masks_queries_logits[-1]

        return [mask_logits], [class_logits]
