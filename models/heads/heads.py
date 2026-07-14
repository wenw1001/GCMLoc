"""
Heads for GCMLoc

介面（符合 GCMLoc_mapping.py）：
    HeatmapHead(in_dim, heatmap_dim)
    Stage1HeatmapDecoder(vmamba_dims, fused_dim, heatmap_dim, target_size)
    PoseRegressor(in_dim, hidden, dropout)

此檔案集中所有 head 模組，供 models.heads.heads 匯入。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PoseRegressor — 純 FC，輸入為 1D 特徵向量
# ---------------------------------------------------------------------------

class PoseRegressor(nn.Module):
    """
    接收空間加權 pooling 後的 1D 特徵向量，回歸平移 t ∈ R³ 與四元數 q ∈ R⁴。

    Args:
        in_dim (int): 輸入特徵維度（等於 heatmap_dim 或 embed_dim）。
        hidden (int): 隱藏層維度，預設 512。
        dropout (float): Dropout 比率。
    """

    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.1)

        self.fc_transl = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden // 2, 3),
        )
        self.fc_rot = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden // 2, 4),
        )

        # 可學習損失權重（對應原版 w_x, w_q）
        self.w_x = nn.Parameter(torch.tensor([0.0]))
        self.w_q = nn.Parameter(torch.tensor([-2.5]))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, in_dim) — 空間加權 pooling 後的特徵向量
        Returns:
            transl: (B, 3)
            rot:    (B, 4)  — 已正規化的單位四元數
        """
        x = self.leaky(self.fc1(x))
        x = self.dropout(x)
        transl = self.fc_transl(x)
        rot = F.normalize(self.fc_rot(x), dim=1)
        return transl, rot


# ---------------------------------------------------------------------------
# HeatmapHead — Stage 2 用（輸入 VMamba 特徵，輸出 heatmap 做加權 pooling）
# ---------------------------------------------------------------------------

class HeatmapHead(nn.Module):
    """
    Stage 2 Heatmap Head。

    接收 VMamba 提取的 GCMLoc 特徵，輸出 softmax 正規化的空間注意力圖，
    供後續加權 pooling 使用。

    Args:
        in_dim (int): 輸入特徵通道數。
        heatmap_dim (int): 輸出通道數（注意力維度）。
    """

    def __init__(self, in_dim: int, heatmap_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, heatmap_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(heatmap_dim, heatmap_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_dim),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, F_M: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_M: (B, in_dim, H', W')
        Returns:
            ch: (B, heatmap_dim, H', W')  — softmax 正規化的空間注意力圖
        """
        B, C, H, W = F_M.shape
        x = self.conv(F_M)                     # (B, heatmap_dim, H', W')
        x = x.reshape(B, x.shape[1], H * W)
        x = self.softmax(x)
        x = x.reshape(B, x.shape[1], H, W)
        return x


# ---------------------------------------------------------------------------
# Stage1HeatmapDecoder — U-Net 式解碼器（融合特徵 → 全解析度 heatmap）
# ---------------------------------------------------------------------------

class Stage1HeatmapDecoder(nn.Module):
    """
    Stage 1 Heatmap 解碼器（U-Net 結構）。

    使用 VMamba 多尺度特徵（skip connections）上採樣融合特徵至目標解析度，
    生成 softmax 正規化的 heatmap，供點選取與加權 pooling 使用。

    Args:
        vmamba_dims (tuple): VMamba 各 stage 通道數 (C1, C2, C3, C4)。
        fused_dim (int): CrossModalFusion 輸出通道數。
        heatmap_dim (int): 輸出通道數。
        target_size (tuple): 目標輸出解析度 (H, W)，通常為原始影像大小。
    """

    def __init__(
        self,
        vmamba_dims: tuple = (96, 192, 384, 768),
        fused_dim: int = 128,
        heatmap_dim: int = 64,
        target_size: tuple = (384, 1280),
    ):
        super().__init__()
        self.target_size = target_size
        C1, C2, C3, C4 = vmamba_dims

        # 輸入投影（fused_dim → C3，C3 為 stage3 = H/16×W/16）
        self.input_proj = nn.Sequential(
            nn.Conv2d(fused_dim, C3, 1, bias=False),
            nn.BatchNorm2d(C3),
            nn.GELU(),
        )

        # 解碼：stage3 → stage2（C3+C2 → C2）
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(C3, C2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C2),
            nn.GELU(),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(C2 + C2, C2, 3, padding=1, bias=False),
            nn.BatchNorm2d(C2),
            nn.GELU(),
        )

        # 解碼：stage2 → stage1（C2+C1 → C1）
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(C2, C1, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C1),
            nn.GELU(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(C1 + C1, C1, 3, padding=1, bias=False),
            nn.BatchNorm2d(C1),
            nn.GELU(),
        )

        # 輸出頭（C1 → heatmap_dim）
        self.head = nn.Sequential(
            nn.Conv2d(C1, heatmap_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(heatmap_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(heatmap_dim, heatmap_dim, 1, bias=True),
        )

        self.softmax = nn.Softmax(dim=-1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, F_fused: torch.Tensor, vmamba_feats: list) -> torch.Tensor:
        """
        Args:
            F_fused:       (B, fused_dim, H', W')  — CrossModalFusion 輸出
            vmamba_feats:  list[Tensor]             — VMamba 各 stage 特徵
                           [f1(H/4), f2(H/8), f3(H/16), f4(H/32)]
        Returns:
            h: (B, heatmap_dim, H_out, W_out)  — softmax 正規化的 heatmap
        """
        f1, f2, f3, f4 = vmamba_feats

        # 投影至 C3 解析度
        x = self.input_proj(F_fused)       # (B, C3, H', W')

        # 確保 x 與 f3 解析度相同
        if x.shape[2:] != f3.shape[2:]:
            x = F.interpolate(x, size=f3.shape[2:], mode="bilinear", align_corners=False)

        # 上採樣 stage3 → stage2
        x = self.up3(x)                    # (B, C2, H/8, W/8)
        if x.shape[2:] != f2.shape[2:]:
            x = F.interpolate(x, size=f2.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec3(torch.cat([x, f2], dim=1))   # (B, C2, H/8, W/8)

        # 上採樣 stage2 → stage1
        x = self.up2(x)                    # (B, C1, H/4, W/4)
        if x.shape[2:] != f1.shape[2:]:
            x = F.interpolate(x, size=f1.shape[2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat([x, f1], dim=1))   # (B, C1, H/4, W/4)

        # 輸出頭
        x = self.head(x)                   # (B, heatmap_dim, H/4, W/4)

        # 上採樣至目標解析度
        if x.shape[2:] != tuple(self.target_size):
            x = F.interpolate(x, size=self.target_size, mode="bilinear", align_corners=False)

        # Softmax 正規化
        B, C, H, W = x.shape
        x = self.softmax(x.reshape(B, C, H * W)).reshape(B, C, H, W)
        return x
