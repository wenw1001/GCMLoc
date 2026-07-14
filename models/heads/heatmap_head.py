"""
Heatmap Head for GCMLoc — Stage 1 (M3 下游)

架構規劃書 §1.2 流程：
    F_fused_off  →  HeatmapHead  →  T_0（Stage 1 heatmap 與初始位姿估計）

功能：
    1. 從融合特徵 F_fused 上採樣生成全解析度 heatmap H ∈ (B, 1, H, W)
    2. softmax 正規化生成機率 heatmap H_hat
    3. 加權平均（期望）生成 2D 峰值座標
    4. 可回傳中間特徵供後續 Stage 2 使用

輸入：
    F_fused: (B, C, H', W')  — 融合特徵（Stage 1）
輸出：
    heatmap_logit:  (B, 1, H_out, W_out)  — 未正規化 logit
    heatmap_prob:   (B, 1, H_out, W_out)  — softmax 正規化
    peak_coords:    (B, 2)  — 期望 2D 峰值座標 (x, y)，範圍 [0, W-1] 和 [0, H-1]
    feat_out:       (B, C_out, H_out, W_out)  — 中間特徵（供 Stage 2）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapHead(nn.Module):
    """
    Stage 1 Heatmap Head。

    Pipeline：
        F_fused (B, C, H', W')
            │
            ▼
        [DeConv 上採樣至目標解析度]
            │
            ▼
        [Conv layers 精煉特徵]
            │
            ├──► feat_out   (B, C_out, H_out, W_out)
            │
            └──► heatmap_logit → softmax → heatmap_prob → peak_coords

    Args:
        in_channels (int): 輸入特徵通道數 C。
        out_channels (int): 中間特徵輸出通道數 C_out（供 Stage 2）。
        output_size (tuple[int, int]): 目標 heatmap 解析度 (H_out, W_out)。
        num_upsample (int): 上採樣倍數（2^n 倍），預設 4（16倍解析度補償）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        output_size: tuple = (384, 1280),
        num_upsample: int = 4,
    ):
        super().__init__()
        self.output_size = output_size

        # --- 上採樣 + 特徵精煉 ---
        layers = []
        cur_ch = in_channels
        # 逐步 2× 上採樣
        for i in range(num_upsample):
            next_ch = max(cur_ch // 2, out_channels)
            layers += [
                nn.ConvTranspose2d(cur_ch, next_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(inplace=True),
            ]
            cur_ch = next_ch

        # 確保最終通道數為 out_channels
        if cur_ch != out_channels:
            layers += [
                nn.Conv2d(cur_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        self.upsample_block = nn.Sequential(*layers)

        # --- Heatmap 生成頭 ---
        self.heatmap_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, 1, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def softmax_heatmap(logit: torch.Tensor) -> torch.Tensor:
        """
        將 heatmap logit 做空間 softmax 正規化。

        Args:
            logit: (B, 1, H, W)
        Returns:
            prob: (B, 1, H, W)，每個 batch 加總為 1
        """
        B, _, H, W = logit.shape
        prob = F.softmax(logit.reshape(B, -1), dim=-1).reshape(B, 1, H, W)
        return prob

    @staticmethod
    def compute_peak_coords(heatmap_prob: torch.Tensor) -> torch.Tensor:
        """
        從機率 heatmap 計算期望 2D 座標（soft-argmax）。

        Args:
            heatmap_prob: (B, 1, H, W)
        Returns:
            coords: (B, 2)，(x, y) 以像素為單位
        """
        B, _, H, W = heatmap_prob.shape
        device = heatmap_prob.device

        # 建立座標網格 [0, W-1] × [0, H-1]
        grid_x = torch.arange(W, dtype=torch.float32, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        grid_y = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1).expand(B, 1, H, W)

        # 期望座標
        exp_x = (heatmap_prob * grid_x).flatten(1).sum(dim=1)  # (B,)
        exp_y = (heatmap_prob * grid_y).flatten(1).sum(dim=1)  # (B,)

        coords = torch.stack([exp_x, exp_y], dim=1)  # (B, 2)
        return coords

    def forward(self, F_fused: torch.Tensor) -> dict:
        """
        Args:
            F_fused: (B, C, H', W')
        Returns:
            dict with keys:
                'heatmap_logit'  : (B, 1, H_out, W_out)
                'heatmap_prob'   : (B, 1, H_out, W_out)
                'peak_coords'    : (B, 2)
                'feat_out'       : (B, C_out, H_out, W_out)
        """
        # 上採樣 + 精煉
        feat = self.upsample_block(F_fused)  # (B, C_out, H_up, W_up)

        # 若上採樣後解析度與目標不符，做最終 resize 對齊
        if feat.shape[2:] != tuple(self.output_size):
            feat = F.interpolate(
                feat, size=self.output_size, mode="bilinear", align_corners=False
            )

        # Heatmap 生成
        heatmap_logit = self.heatmap_conv(feat)          # (B, 1, H_out, W_out)
        heatmap_prob = self.softmax_heatmap(heatmap_logit)
        peak_coords = self.compute_peak_coords(heatmap_prob)

        return {
            "heatmap_logit": heatmap_logit,
            "heatmap_prob": heatmap_prob,
            "peak_coords": peak_coords,
            "feat_out": feat,
        }
