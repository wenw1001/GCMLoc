"""
VMamba Backbone for GCMLoc (M2)

架構規劃書 §3 — VMamba 骨幹 + 多輸入分支。

有三個後端可選（自動偵測）：
  1. 官方 VMamba (vmamba.models.vmamba.VSSBlock)   — 需要手動 pip install -e VMamba/
  2. mamba-ssm Mamba (推薦，pip install mamba-ssm) — 真正的 SSM，有 selective_scan CUDA kernel
  3. CNN Fallback (DW-Conv + FFN)                  — 純 PyTorch，無需 CUDA extensions

介面（符合 GCMLoc_mapping.py）：
    VMambaBackbone(out_dim, vmamba_dims, output_stage, in_channels, use_cnn_fallback)
    .extract_depth_gt(depth)   → (F_D_gt, multi_scale_feats)
    .extract_depth_init(lidar) → F_D_init
    .extract_gcmloc(gcmloc)      → (F_M, multi_scale_feats)

use_cnn_fallback=True：強制使用 CNN 近似（忽略已安裝的 SSM 套件）
use_cnn_fallback=False：優先嘗試官方 VMamba → mamba-ssm Mamba → CNN
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 後端偵測（按優先順序）
# ---------------------------------------------------------------------------

_BACKEND = "cnn"   # 預設 fallback
_OfficialVSSBlockCls = None   # 快取官方 VSSBlock class

def _detect_backend():
    global _BACKEND, _OfficialVSSBlockCls

    # 1. 嘗試本地 VMamba clone（優先：models/VMamba/vmamba.py）
    import sys, os
    _local_vmamba = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'models', 'VMamba'
    )
    if os.path.exists(os.path.join(_local_vmamba, 'vmamba.py')):
        if _local_vmamba not in sys.path:
            sys.path.insert(0, _local_vmamba)
        try:
            from vmamba import VSSBlock as _V
            _OfficialVSSBlockCls = _V
            _BACKEND = "vmamba_official"
            print("use local VMamba")
            return
        except Exception as e:
            # vmamba 載入失敗（例如缺少 selective_scan 算子），移除路徑並繼續
            if _local_vmamba in sys.path:
                sys.path.remove(_local_vmamba)

    # 2. 嘗試已安裝的 vmamba 套件
    try:
        from vmamba.models.vmamba import VSSBlock as _V
        _OfficialVSSBlockCls = _V
        _BACKEND = "vmamba_official"
        print("use installed VMamba")
        return
    except ImportError:
        pass

    # 3. 嘗試 mamba-ssm
    try:
        from mamba_ssm import Mamba as _M   # noqa: F401
        _BACKEND = "mamba_ssm"
        return
    except ImportError:
        pass

    _BACKEND = "cnn"


# ---------------------------------------------------------------------------
# 各後端 VSS Block 實作
# ---------------------------------------------------------------------------

class _CnnVSSBlock(nn.Module):
    """CNN 近似（DW-Conv + FFN），純 PyTorch，無需 CUDA extensions。"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dw = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        r = x
        xn = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = r + self.dw(xn)
        r = x
        flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x = r + self.norm2(flat).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = r + self.ffn(x.permute(0, 2, 3, 1).reshape(B * H * W, C)).reshape(
            B, H, W, C
        ).permute(0, 3, 1, 2)
        return x


class _MambaSSMBlock(nn.Module):
    """
    使用 mamba_ssm.Mamba 的真正 SSM block。

    Mamba 的輸入格式是 (B, L, C)（sequence），
    這裡把 2D feature map 壓成 sequence（行掃描）再還原。
    """
    def __init__(self, dim: int):
        super().__init__()
        from mamba_ssm import Mamba
        self.norm = nn.LayerNorm(dim)
        # d_model=dim, d_state=16, d_conv=4, expand=2（Mamba 預設）
        self.mamba = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 2D → sequence: (B, H*W, C)
        seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        seq = self.norm(seq)
        out = self.mamba(seq)                       # (B, H*W, C)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out                              # residual connection


class _OfficialVSSBlock(nn.Module):
    """官方 VMamba VSSBlock 包裝器（支援 models/VMamba/vmamba.py 本地 clone）。"""
    def __init__(self, dim: int):
        super().__init__()
        if _OfficialVSSBlockCls is None:
            raise RuntimeError("VSSBlock class not loaded")
        # VMamba VSSBlock 介面：hidden_dim, drop_path, channel_first=False
        self.block = _OfficialVSSBlockCls(
            hidden_dim=dim,
            drop_path=0.0,
            channel_first=False,
            forward_type="v02"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VMamba VSSBlock 輸入格式：(B, H, W, C)（channel_first=False）
        B, C, H, W = x.shape
        x_bhwc = x.permute(0, 2, 3, 1)          # (B, H, W, C)
        out = self.block(x_bhwc)                 # (B, H, W, C)
        return out.permute(0, 3, 1, 2)           # (B, C, H, W)


def _make_vss_block(dim: int, use_cnn_fallback: bool) -> nn.Module:
    """根據後端設定建立對應的 VSS Block。"""
    if use_cnn_fallback:
        return _CnnVSSBlock(dim)
    if _BACKEND == "vmamba_official":
        return _OfficialVSSBlock(dim)
    if _BACKEND == "mamba_ssm":
        return _MambaSSMBlock(dim)
    return _CnnVSSBlock(dim)


# ---------------------------------------------------------------------------
# 基礎組件
# ---------------------------------------------------------------------------

class _PatchMerging(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.conv(x)


class _Stage(nn.Module):
    def __init__(self, dim: int, num_blocks: int, use_cnn_fallback: bool):
        super().__init__()
        self.blocks = nn.Sequential(
            *[_make_vss_block(dim, use_cnn_fallback) for _ in range(num_blocks)]
        )
    def forward(self, x): return self.blocks(x)


class _InputHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x): return self.proj(x)


# ---------------------------------------------------------------------------
# 共享骨幹內部實作
# ---------------------------------------------------------------------------

class _SharedVMambaNet(nn.Module):
    def __init__(self, dims: tuple, num_blocks: tuple, use_cnn_fallback: bool, output_stage: int):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv2d(dims[0], dims[0], kernel_size=4, stride=4, padding=0, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        self.stage1 = _Stage(dims[0], num_blocks[0], use_cnn_fallback)
        self.down1   = _PatchMerging(dims[0], dims[1])
        self.stage2  = _Stage(dims[1], num_blocks[1], use_cnn_fallback)
        self.down2   = _PatchMerging(dims[1], dims[2])
        self.stage3  = _Stage(dims[2], num_blocks[2], use_cnn_fallback)
        self.down3   = _PatchMerging(dims[2], dims[3])
        self.stage4  = _Stage(dims[3], num_blocks[3], use_cnn_fallback)

        self.output_stage = output_stage   # 1~4

    def forward(self, x: torch.Tensor):
        f1 = self.stage1(self.patch_embed(x))    # (B, C1, H/4,  W/4)
        f2 = self.stage2(self.down1(f1))          # (B, C2, H/8,  W/8)
        f3 = self.stage3(self.down2(f2))          # (B, C3, H/16, W/16)
        f4 = self.stage4(self.down3(f3))          # (B, C4, H/32, W/32)

        feats = [f1, f2, f3, f4]
        primary = feats[self.output_stage - 1]
        return primary, feats


# ---------------------------------------------------------------------------
# 對外介面：VMambaBackbone
# ---------------------------------------------------------------------------

class VMambaBackbone(nn.Module):
    """
    VMamba 骨幹（符合 GCMLoc_mapping.py 介面）。

    後端選擇優先順序（use_cnn_fallback=False 時）：
        官方 VMamba > mamba-ssm Mamba > CNN fallback

    Args:
        out_dim (int): 投影輸出通道數（對齊 DINOv2 的 out_dim）。
        vmamba_dims (tuple): 各 stage 通道數，預設 (96, 192, 384, 768)。
        output_stage (int): 主要輸出 stage（1~4）。
        in_channels (int): 深度圖輸入通道數（通常 1）。
        use_cnn_fallback (bool): True=強制使用 CNN；False=自動選最佳後端。
        num_blocks (tuple): 各 stage 的 SSM Block 數量。
    """

    def __init__(
        self,
        out_dim: int = 128,
        vmamba_dims: tuple = (96, 192, 384, 768),
        output_stage: int = 2,
        in_channels: int = 1,
        use_cnn_fallback: bool = True,
        num_blocks: tuple = (2, 2, 6, 2),
    ):
        super().__init__()
        self.out_dim = out_dim

        _detect_backend()

        # 後端選擇邏輯
        actual_fallback = use_cnn_fallback
        if not use_cnn_fallback and _BACKEND == "cnn":
            warnings.warn(
                "use_cnn_fallback=False 但找不到官方 VMamba 或 mamba-ssm，"
                "自動退到 CNN fallback。\n"
                "安裝方式：pip install mamba-ssm",
                UserWarning,
            )
            actual_fallback = True

        backend_name = "CNN-fallback" if actual_fallback else _BACKEND
        print(f"VMamba backend: {backend_name}")

        C0 = vmamba_dims[0]

        # --- 各輸入分支的 InputHead（不同用途，獨立權重）---
        self.head_depth_gt   = _InputHead(in_channels, C0)   # D_gt（GT 深度）
        self.head_depth_init = _InputHead(in_channels, C0)   # D_init（初始深度）
        self.head_lhmap      = _InputHead(in_channels, C0)   # GCMLoc (M^r)

        # --- 共享 VMamba 骨幹 ---
        self.backbone = _SharedVMambaNet(
            dims=vmamba_dims,
            num_blocks=num_blocks,
            use_cnn_fallback=actual_fallback,
            output_stage=output_stage,
        )

        # --- 各分支投影層（primary stage 輸出 → out_dim）---
        primary_dim = vmamba_dims[output_stage - 1]
        self.proj_depth_gt   = self._make_proj(primary_dim, out_dim)
        self.proj_depth_init = self._make_proj(primary_dim, out_dim)
        self.proj_lhmap      = self._make_proj(primary_dim, out_dim)

        self._init_weights()

    @staticmethod
    def _make_proj(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
            nn.GELU(),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def extract_depth_gt(self, depth: torch.Tensor):
        """
        處理 GT 深度圖 D_gt（Stage 1 幾何特徵）。

        Args:
            depth: (B, 1, H, W) — 建議先做 log(1+D) 或 D/D_max 正規化
        Returns:
            F_D_gt:  (B, out_dim, H', W')
            feats:   list[Tensor] — 各 stage 特徵供 U-Net skip connection
        """
        x = self.head_depth_gt(depth)
        primary, feats = self.backbone(x)
        F_D_gt = self.proj_depth_gt(primary)
        return F_D_gt, feats

    def extract_depth_init(self, lidar: torch.Tensor):
        """
        處理初始投影深度 D_init（Stage 1 相關性計算）。

        Args:
            lidar: (B, 1, H, W)
        Returns:
            F_D_init: (B, out_dim, H', W')
            feats:    list[Tensor] — 各 stage 特徵供多尺度 Flow Matcher 使用
        """
        x = self.head_depth_init(lidar)
        primary, feats = self.backbone(x)
        return self.proj_depth_init(primary), feats

    def extract_gcmloc(self, gcmloc: torch.Tensor):
        """
        處理 GCMLoc（M^r），Stage 2 地圖特徵。

        Args:
            gcmloc: (B, K, H, W)，K=1 或多通道
        Returns:
            F_M:   (B, out_dim, H', W')
            feats: list[Tensor] — 各 stage 特徵供 U-Net skip connection
        """
        x = self.head_lhmap(gcmloc)
        primary, feats = self.backbone(x)
        F_M = self.proj_lhmap(primary)
        return F_M, feats
