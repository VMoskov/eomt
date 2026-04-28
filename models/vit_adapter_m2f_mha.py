# ---------------------------------------------------------------
# MHA variant of vit_adapter_m2f.py — for memory benchmarking only.
# Encoder uses standard MHA (nn.MultiheadAttention) in interaction
# blocks instead of MSDeformAttn. Decoder is unchanged.
# ---------------------------------------------------------------

import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
    Mask2FormerPixelDecoderEncoderMultiscaleDeformableAttention as MSDeformAttn,
    Mask2FormerTransformerModule,
)

from models.vit_adapter_mha import ViTAdapterMHAEncoder


class ViTAdapterMHAM2F(nn.Module):
    """
    ViT-Adapter (MHA) encoder + HuggingFace Mask2Former decoder.
    Use instead of ViTAdapterM2F to measure the memory cost of MHA
    vs MSDeformAttn in the interaction blocks.
    """

    masked_attn_enabled: bool = False
    num_blocks: int = 1

    def __init__(
        self,
        backbone_name: str = "vit_large_patch14_reg4_dinov2",
        img_size: tuple = (640, 640),
        num_classes: int = 133,
        num_queries: int = 200,
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
        feature_size: int = 256,
        hidden_dim: int = 256,
        encoder_layers: int = 6,
        decoder_layers: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoder = ViTAdapterMHAEncoder(
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

        embed_dim = self.encoder.backbone.embed_dim

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

        self.pixel_decoder = Mask2FormerPixelDecoder(
            config, feature_channels=[embed_dim] * 4
        )

        self.transformer_module = Mask2FormerTransformerModule(
            in_features=feature_size, config=config
        )

        self.class_predictor = nn.Linear(hidden_dim, num_classes + 1)

        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Fix HF's standalone init gaps: uninitialized level_embed and Kaiming MSDeformAttn."""
        nn.init.zeros_(self.pixel_decoder.level_embed)

        for module in self.pixel_decoder.modules():
            if not isinstance(module, MSDeformAttn):
                continue
            nn.init.constant_(module.sampling_offsets.weight, 0.)
            thetas = torch.arange(module.n_heads, dtype=torch.float32) * (2.0 * math.pi / module.n_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
            grid_init = grid_init / grid_init.abs().max(dim=-1, keepdim=True).values
            grid_init = grid_init.view(module.n_heads, 1, 1, 2).repeat(1, module.n_levels, module.n_points, 1)
            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1
            with torch.no_grad():
                module.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            nn.init.constant_(module.attention_weights.weight, 0.)
            nn.init.constant_(module.attention_weights.bias, 0.)
            nn.init.xavier_uniform_(module.value_proj.weight)
            nn.init.constant_(module.value_proj.bias, 0.)
            nn.init.xavier_uniform_(module.output_proj.weight)
            nn.init.constant_(module.output_proj.bias, 0.)

    def forward(self, x: torch.Tensor):
        x = x.float()
        x = (x - self.pixel_mean) / self.pixel_std

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            features = self.encoder(x)
            pix_out = self.pixel_decoder(features)
            trans_out = self.transformer_module(
                multi_scale_features=list(pix_out.multi_scale_features),
                mask_features=pix_out.mask_features,
                output_hidden_states=False,
            )
            class_logits = self.class_predictor(trans_out.last_hidden_state)
            mask_logits = trans_out.masks_queries_logits[-1]

        return [mask_logits], [class_logits]
