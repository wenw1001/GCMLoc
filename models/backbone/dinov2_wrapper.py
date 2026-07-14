"""
DINOv2 Frozen Backbone for GCMLoc (M1)

架構規劃書 §2 — DINOv2 ViT-B/14 凍結骨幹 + 可訓練投影層。

匯出類別：
    DINOv2Backbone  — GCMLoc_mapping 使用的主要介面
    DINOv2Wrapper   — 相容舊有程式碼的別名

介面：
    DINOv2Backbone(out_dim=128, variant='b')
    variant: 'b' → dinov2_vitb14 (768-dim), 's' → dinov2_vits14 (384-dim)
    forward(rgb: Tensor) → F_I: (B, out_dim, 37, 37)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ViT-S/14: 384-dim;  ViT-B/14: 768-dim
_VARIANT_CFG = {
    's': ('dinov2_vits14', 384),
    'b': ('dinov2_vitb14', 768),
}


class DINOv2Backbone(nn.Module):
    """
    凍結 DINOv2 骨幹 + 可訓練投影層。

    Args:
        out_dim (int): 投影輸出通道數 C，建議 128 或 256。
        variant (str): 'b' (ViT-B/14, 768-dim) 或 's' (ViT-S/14, 384-dim)。
        dino_input_size (int): DINOv2 輸入解析度（需為 14 的倍數），預設 518。
        use_intermediate_layers (list[int] | None):
            若指定（如 [4, 8, 12]）則提取中間層特徵並 concat 後投影；
            None 則只用最後一層 patch token。
    """

    def __init__(
        self,
        out_dim: int = 128,
        variant: str = 'b',
        dino_input_size: int = 518,
        use_intermediate_layers: list = None,
    ):
        super().__init__()
        if variant not in _VARIANT_CFG:
            raise ValueError(f"DINOv2Backbone: unknown variant '{variant}', choose 's' or 'b'")
        hub_name, dino_dim = _VARIANT_CFG[variant]

        self.out_dim = out_dim
        self.variant = variant
        self.DINO_DIM = dino_dim
        self.dino_input_size = dino_input_size
        self.use_intermediate_layers = use_intermediate_layers

        # --- 載入 DINOv2 並完全凍結 ---
        self.dino = torch.hub.load(
            "facebookresearch/dinov2",
            hub_name,
            pretrained=True,
        )
        for param in self.dino.parameters():
            param.requires_grad = False

        # --- 可訓練投影層 Conv1x1(DINO_DIM → out_dim) → GroupNorm → GELU ---
        in_dim = (
            dino_dim * len(use_intermediate_layers)
            if use_intermediate_layers is not None
            else dino_dim
        )
        self.projection = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_dim),   # 等效 LayerNorm，相容任意空間尺寸
            nn.GELU(),
        )

        self.patch_h = dino_input_size // 14
        self.patch_w = dino_input_size // 14

        self._init_weights()

    def _init_weights(self):
        for m in self.projection.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")

    def _dino_trainable(self) -> bool:
        """是否有任何 DINOv2 參數被解凍（決定提取時是否建立 autograd 圖）。"""
        return any(p.requires_grad for p in self.dino.parameters())

    def _extract_raw(self, x: torch.Tensor) -> torch.Tensor:
        """
        從 DINOv2 提取 patch token。

        梯度開關由 forward() 依解凍狀態以條件式 no_grad 控制，本函式本身不再強制
        no_grad，以便解凍的 transformer blocks 能正常接收梯度進行微調。

        Returns:
            (B, DINO_DIM[*n], patch_h, patch_w)
        """
        if self.use_intermediate_layers is not None:
            feats = self.dino.get_intermediate_layers(
                x, n=self.use_intermediate_layers, return_class_token=False
            )
            feat = torch.cat(feats, dim=-1)        # (B, N_p, DIM*n)
        else:
            out = self.dino.forward_features(x)
            feat = out["x_norm_patchtokens"]        # (B, N_p, 768)

        B, N_p, C = feat.shape
        feat = feat.permute(0, 2, 1).reshape(B, C, self.patch_h, self.patch_w)
        return feat

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) — 原始 RGB 影像
        Returns:
            F_I: (B, out_dim, patch_h, patch_w)
        """
        # Resize 至 DINOv2 要求大小
        if rgb.shape[-2:] != (self.dino_input_size, self.dino_input_size):
            rgb = F.interpolate(
                rgb,
                size=(self.dino_input_size, self.dino_input_size),
                mode="bilinear",
                align_corners=False,
            )

        # 條件式 no_grad：有解凍 blocks 時建圖以支援微調；完全凍結時關閉以省記憶體。
        # （外層若已在 torch.no_grad() 下，如驗證階段，仍會被尊重）
        if self._dino_trainable():
            feat_raw = self._extract_raw(rgb)       # (B, DINO_DIM, 37, 37)，建立 autograd 圖
        else:
            with torch.no_grad():
                feat_raw = self._extract_raw(rgb)   # 全凍結，省記憶體

        F_I = self.projection(feat_raw)     # (B, out_dim, 37, 37)
        return F_I

    def get_output_spatial_size(self) -> tuple:
        return (self.patch_h, self.patch_w)


# 向後相容別名
DINOv2Wrapper = DINOv2Backbone
