from typing import Any, Dict, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d

from model.base.base_encoder_ref import BaseEncoderRef


class ConvLNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=None, act=nn.GELU):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False
        )
        self.ln = LayerNorm2d(out_ch)
        self.act = act() if act is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        x = self.act(x)
        return x


class ResNetBasicBlockLN(nn.Module):
    def __init__(self, ch, act=nn.GELU):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.ln1 = LayerNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1, bias=False)
        self.ln2 = LayerNorm2d(ch)
        self.act = act()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.ln2(out)
        out = out + identity
        out = self.act(out)
        return out


class PixelwiseMLP(nn.Module):
    def __init__(self, ch, hidden_mul=2, act=nn.GELU):
        super().__init__()
        self.net = nn.Sequential(
            ConvLNAct(ch, hidden_mul * ch, k=1, s=1, act=act),
            ConvLNAct(hidden_mul * ch, ch, k=1, s=1, act=act),
        )

    def forward(self, x):
        return self.net(x)


class ContextPooling(nn.Module):
    def __init__(self, ch, act=nn.GELU):
        super().__init__()
        self.mlp1 = ConvLNAct(ch, ch, k=1, s=1, act=act)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            LayerNorm2d(ch),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, 0, bias=False),
            LayerNorm2d(ch),
        )
        self.post = LayerNorm2d(ch)

    def forward(self, x):
        b, c, h, w = x.shape
        g = F.adaptive_avg_pool2d(x, 1)  # B,C,1,1
        g = self.mlp1(g)  # B,C,1,1
        g = self.mlp2(g)  # B,C,1,1
        g = g.expand(-1, -1, h, w)  # broadcast
        x_proj = self.proj(x)  # B,C,H,W
        out = x_proj + g
        out = self.post(out)
        return out


def up_to(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size_hw:
        return x
    return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)


class MultiScaleSumFusion(nn.Module):
    """
    - Lateral 1x1 auf c Kanäle für alle Stufen
    - Upsample auf target_size
    - Summe
    """

    def __init__(self, in_channels: Sequence[int], out_ch: int, act=nn.GELU):
        super().__init__()
        self.lateral = nn.ModuleList(
            [ConvLNAct(c_in, out_ch, k=1, s=1, act=act) for c_in in in_channels]
        )

    def forward(
        self, feats: Sequence[torch.Tensor], target_hw: Tuple[int, int]
    ) -> torch.Tensor:
        assert len(feats) == len(self.lateral)
        y = None
        for f, lat in zip(feats, self.lateral):
            f = lat(f)
            f = up_to(f, target_hw)
            y = f if y is None else (y + f)
        return y


class ReferenceEncoder(BaseEncoderRef):
    """
    ConvNeXt (features_only) + zwei ResNet-ähnliche Blöcke @stride=1 auf dem Eingabebild
    -> ContextPooling auf C4
    -> Multi-scale Sum über [C1,C2,C3,C4_cp, S1_extra]
    -> Upsample auf sA=1 (HxW)
    -> Pixelwise MLP
    -> optional L2-Norm
    Ausgabe: BxCxHxW
    """

    def __init__(
        self,
        variant: str = "convnext_tiny",
        out_channels: int = 8,
        context_pooling: bool = True,
        pretrained: bool = True,
        in_chans: int = 3,
        mlp_hidden_mul: int = 2,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            variant,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # strides 4,8,16,32
            pretrained=pretrained,
            in_chans=in_chans,
        )
        in_channels = self.backbone.feature_info.channels()

        # Pfad @ stride 1 direkt auf dem Eingabebild: stem -> 2 ResNet-BasicBlocks
        self.stem = ConvLNAct(in_chans, out_channels, k=3, s=1, p=1)  # 3x3, stride=1
        self.res1 = ResNetBasicBlockLN(out_channels)
        self.res2 = ResNetBasicBlockLN(out_channels)

        # ContextPooling auf C4
        self.context = (
            ContextPooling(in_channels[-1]) if context_pooling else nn.Identity()
        )

        # Multi-Scale-Fusion über 5 Quellen: C1..C4 + S1
        self.fuse_backbone = MultiScaleSumFusion(in_channels, out_channels)
        # Für S1 (bereits out_channels) nur Upsample + Summe mit dem
        # Backbone-Fusionsergebnis
        self.pixel_mlp = PixelwiseMLP(out_channels, hidden_mul=mlp_hidden_mul)

    def forward_impl(self, x: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        H, W = x["imgs_ref"].shape[-2:]

        # Backbone-Features
        feats = self.backbone(x)

        c1, c2, c3, c4 = feats
        c4 = self.context(c4)

        s1 = self.stem(x)
        s1 = self.res1(s1)
        s1 = self.res2(s1)

        fused_bb = self.fuse_backbone([c1, c2, c3, c4], target_hw=(H, W))  # B,C,H,W
        y = fused_bb + s1
        y = self.pixel_mlp(y)

        return {"feat_ref": y, "gsd_ref": x["gsd"]}  # not changing the gsd
