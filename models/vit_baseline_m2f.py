# ---------------------------------------------------------------
# © 2025. Licensed under the MIT License.
#
# ViTPyramid + HuggingFace Mask2Former for universal segmentation.
# Plugs into EoMT's LightningModule without any changes to existing code.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
    Mask2FormerTransformerModule,
)

from models.vit_pyramid import ViTPyramid


class ViTBaselineM2F(nn.Module):
    """
    ViTPyramid encoder + HuggingFace Mask2Former decoder.

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
        patch_size: int = 14,
        num_classes: int = 133,
        num_queries: int = 200,
        feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ---- Encoder: ViTPyramid ----
        # Handles ImageNet normalisation internally; exposes self.backbone for LLRD.
        self.encoder = ViTPyramid(
            backbone_name=backbone_name,
            img_size=img_size,
            patch_size=patch_size,
        )

        embed_dim = self.encoder.backbone.embed_dim

        # ---- M2F decoder config ----
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
        self.transformer_module = Mask2FormerTransformerModule(
            in_features=feature_size, config=config
        )

        # ---- Class prediction head ----
        self.class_predictor = nn.Linear(hidden_dim, num_classes + 1)

    @torch.compiler.disable()
    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, H, W) float32 in [0, 1]  — normalisation handled by ViTPyramid

        Returns:
          ([mask_logits], [class_logits])
          mask_logits:  (B, num_queries, H/4, W/4)
          class_logits: (B, num_queries, num_classes+1)
        """
        # ViTPyramid encoder runs in fp16 (AMP handles this)
        features = self.encoder(x.float())  # [f4, f8, f16, f32]

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
