"""
DINOv2 RGB backbone for GCMLoc.

    DINOv2Backbone(out_dim=128, variant='b', unfreeze_last_blocks=4)
        variant 'b' -> dinov2_vitb14 (768-d patch tokens, GCMLoc-B)
        variant 's' -> dinov2_vits14 (384-d patch tokens, GCMLoc-S)
        forward(rgb) -> F_I (B, out_dim, 37, 37)

The image is resized to 518x518 (37x37 patches of 14 px), encoded by the ViT and
projected with a 1x1 conv + GroupNorm + GELU to the unified feature dimension.
All ViT blocks are frozen except the last `unfreeze_last_blocks` blocks and the
final LayerNorm.

state_dict layout (unchanged from the trained checkpoints):
    dino.*            DINOv2 ViT (torch.hub facebookresearch/dinov2)
    projection.0/1    Conv1x1 / GroupNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_VARIANT_CFG = {
    's': ('dinov2_vits14', 384),
    'b': ('dinov2_vitb14', 768),
}


class DINOv2Backbone(nn.Module):
    def __init__(self, out_dim: int = 128, variant: str = 'b',
                 unfreeze_last_blocks: int = 4, dino_input_size: int = 518):
        super().__init__()
        if variant not in _VARIANT_CFG:
            raise ValueError(f"DINOv2Backbone: unknown variant {variant!r}, choose 's' or 'b'")
        hub_name, dino_dim = _VARIANT_CFG[variant]

        self.out_dim = out_dim
        self.variant = variant
        self.dino_dim = dino_dim
        self.dino_input_size = dino_input_size
        self.patch_h = self.patch_w = dino_input_size // 14

        self.dino = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=True)
        for p in self.dino.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Conv2d(dino_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_dim),
            nn.GELU(),
        )
        nn.init.kaiming_normal_(self.projection[0].weight, mode="fan_out")

        self.n_unfrozen = self._unfreeze_last_blocks(unfreeze_last_blocks)

    # ------------------------------------------------------------------
    def _unfreeze_last_blocks(self, n_blocks: int) -> int:
        """Make the last `n_blocks` transformer blocks (+ final LayerNorm) trainable."""
        total = len(self.dino.blocks)
        n = max(0, min(n_blocks, total))
        for block in self.dino.blocks[total - n:]:
            for p in block.parameters():
                p.requires_grad = True
            # self._reinit_linear(block)
        if n > 0 and hasattr(self.dino, 'norm'):
            for p in self.dino.norm.parameters():
                p.requires_grad = True
        return n

    @staticmethod
    def _reinit_linear(module: nn.Module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

    # ------------------------------------------------------------------
    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.dino.forward_features(x)["x_norm_patchtokens"]       # (B, N, dino_dim)
        B, N, C = feat.shape
        return feat.permute(0, 2, 1).reshape(B, C, self.patch_h, self.patch_w)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """rgb (B, 3, H, W) -> F_I (B, out_dim, 37, 37)"""
        if rgb.shape[-2:] != (self.dino_input_size, self.dino_input_size):
            rgb = F.interpolate(rgb, size=(self.dino_input_size, self.dino_input_size),
                                mode="bilinear", align_corners=False)
        if self.n_unfrozen > 0:
            feat = self._extract_tokens(rgb)            # keep the graph for fine-tuning
        else:
            with torch.no_grad():
                feat = self._extract_tokens(rgb)
        return self.projection(feat)
