from typing import Any, Dict, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import LayerNorm2d
from transformers import SegformerConfig
from transformers.models.segformer.modeling_segformer import SegformerLayer

from model.base.base_encoder_qry import BaseEncoderQry
from utils.point_pillars import get_points_3d, get_value_tokens


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
    def __init__(self, ch, hidden_mul=4, act=nn.GELU):
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


class GroundEncoder(nn.Module):
    """
    ConvNeXt (timm features_only) -> ContextPooling auf C4 -> Multi-scale Sum auf sG=4
    -> Pixelwise MLP -> optional L2-Norm
    Gibt final BxCx(H/4)x(W/4) zurück.
    """

    def __init__(
        self,
        variant: str = "convnext_tiny",
        out_channels: int = 128,
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
        self.context = (
            ContextPooling(in_channels[-1]) if context_pooling else nn.Identity()
        )
        self.fuse = MultiScaleSumFusion(in_channels, out_channels)
        self.pixel_mlp = PixelwiseMLP(out_channels, hidden_mul=mlp_hidden_mul)
        self.debug = False

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        if self.debug:
            print("Using debug mode in GroundEncoder")
            return x
        H, W = x.shape[-2:]
        feats = self.backbone(x)  # [C1 (H/4), C2 (H/8), C3 (H/16), C4 (H/32)]
        c1, c2, c3, c4 = feats
        c4 = self.context(c4)
        # Zielgröße: sG=4 -> (H/4, W/4)
        target = (H // 4, W // 4)
        y = self.fuse([c1, c2, c3, c4], target_hw=target)
        y = self.pixel_mlp(y)
        return y


def mask_to_max_circle(img: torch.Tensor) -> torch.Tensor:
    """
    Zero out pixels outside the maximum inscribed circle.

    Args:
        img: Tensor of shape (B, C, H, W)

    Returns:
        masked_img: Tensor of shape (B, C, H, W)
    """
    B, C, H, W = img.shape

    # circle radius = half of the smaller dimension
    radius = min(H, W) / 2.0

    # coordinate grid (centered at image center)
    y = torch.arange(H, device=img.device).float() - (H - 1) / 2.0
    x = torch.arange(W, device=img.device).float() - (W - 1) / 2.0
    Y, X = torch.meshgrid(y, x, indexing="ij")

    dist2 = X**2 + Y**2
    mask = (dist2 <= radius**2).to(img.dtype)  # (H, W)

    # expand to (B, C, H, W)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    masked_img = img * mask

    return masked_img


class PVToBEV(nn.Module):
    def __init__(
        self,
        dim=128,
        blocks=3,
        z_points=16,
        h_min=-5,
        h_max=10,
        bev_shape=(40, 40),
        matching_shape=(320, 320),
    ):
        super().__init__()
        self.blocks = blocks
        self.z_points = z_points
        self.h_min = h_min
        self.h_max = h_max
        self.matching_shape = matching_shape

        self.bev = nn.Parameter(
            torch.randn(1, dim, bev_shape[0], bev_shape[1]), requires_grad=True
        )

        self.refinement_blocks = nn.ModuleList(
            [TransformerBlock(dim) for _ in range(blocks)]
        )

        self.out_proj = nn.Conv2d(dim, 8, 1, 1, 0)
        self.debug = False

    def forward(
        self,
        fmaps_ground: torch.Tensor,
        extr: torch.Tensor,
        intr: torch.Tensor,
        res: torch.Tensor,
        m_per_pixel,
    ):
        b, n, _, _, _ = fmaps_ground.shape
        query = self.bev.expand(b, -1, -1, -1)

        points = get_points_3d(
            query,
            m_per_pixel[0].item(),
            upsampling_steps=3,  # 3
            z_min=self.h_min,
            z_max=self.h_max,
            z_points=self.z_points,
        )

        # check if they are same with the original points
        mv_tokens, _ = get_value_tokens(
            points,
            fmaps_ground,
            extr,
            intr,
            res,
        )

        logits_shortcut = None
        for block in self.refinement_blocks:
            # print(f"Using block {block} in PVToBEV")
            query, logits_shortcut = block(query, mv_tokens, logits_shortcut)

        # upsampling to 320, 320 and projection to 8 channels
        query = nn.functional.interpolate(
            query, size=self.matching_shape, mode="bilinear", align_corners=False
        )

        if self.debug:
            print("using debugg mode in PVToBEV")
            return mask_to_max_circle((query))
        return mask_to_max_circle(self.out_proj(query))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        # self.cross_attn = CrossViewAttention(dim, heads)
        self.cross_attn = PointAttention(dim)
        self.self_attn = SegFormerSelfAttention(dim)

        self.norm2 = nn.GroupNorm(1, dim)

        self.mlp = nn.Sequential(
            nn.Conv2d(dim, int(dim * 4), 1),
            nn.GELU(),
            nn.Conv2d(int(dim * 4), dim, 1),
        )
        self.debug = False

    def forward(self, q, mv_token, logits_shortcut=None):
        if self.debug:
            print("Using debug mode in TransformerBlock")
            return mv_token[:, :, :, 0, :].permute(0, 3, 1, 2), None
        q_skip_1 = q
        q = self.norm1(q)
        q, logits = self.cross_attn(q, mv_token, logits_shortcut)  # (B, D, H, W)

        q = self.self_attn(q)  # (B, D, H, W)
        q = q + q_skip_1
        q_skip_2 = q
        q = self.norm2(q)
        q = self.mlp(q) + q_skip_2

        return q, logits


class PointAttention(nn.Module):
    def __init__(self, dim):
        """
        dim:      Feature-Dim der Queries (D)
        dim:      Feature-Dim der Values (D oder anderes)
        num_tokens: Anzahl T der MV-Tokens
        """
        super().__init__()
        # Projektion von q -> attention logits über T
        self.to_logits = None  # wird lazy beim ersten Forward gebaut
        self.dim = dim

    def forward(self, q, mv_token, logits_shortcut=None):
        """
        q:        (B, D, H, W)
        mv_token: (B, H, W, T, D)
        logits_shortcut: (B, (H, W), T) (optional, z.B. aus vorherigem Layer)
        """
        _, _, H, W = q.shape
        T = mv_token.shape[3]  # Anzahl Tokens aus mv_token

        # Falls to_logits noch nicht existiert → jetzt bauen
        if self.to_logits is None:
            self.to_logits = nn.Linear(self.dim, T).to(q.device)

        # Flatten spatial dims
        q_flat = rearrange(q, "b d h w -> b (h w) d")  # (B, N, D), N=H*W
        # Attention logits direkt aus q
        logits = self.to_logits(q_flat)  # (B, N, T)

        # optionaler Shortcut
        if logits_shortcut is not None:
            logits = logits + logits_shortcut

        # Attention-Weights
        attn = torch.softmax(logits, dim=-1)  # (B, N, T)

        # Prepare Values
        v = rearrange(mv_token, "b h w t d -> b (h w) t d")  # (B, N, T, Dv)

        # Weighted sum über Tokens
        out = torch.einsum("bnt, bntd -> bnd", attn, v)  # (B, N, D)

        # zurück in (B, D, H, W)
        out = rearrange(out, "b (h w) d -> b d h w", h=H, w=W)
        return out, logits


class SegFormerSelfAttention(nn.Module):
    def __init__(self, dim, sr_ratio=4):
        super().__init__()
        self.self_attn = SegformerLayer(
            config=SegformerConfig(),
            hidden_size=dim,
            num_attention_heads=1,
            mlp_ratio=4,
            sequence_reduction_ratio=sr_ratio,
            drop_path=0.0,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        y_seq = self.self_attn(x_seq, H, W)[0]
        y_img = y_seq.transpose(1, 2).reshape(B, C, H, W)

        return y_img


class QueryEncoder(BaseEncoderQry):
    def __init__(self, args):
        super().__init__(args)
        self.ground_encoder = GroundEncoder(variant="convnext_base")
        self.PVToBEV = PVToBEV(z_points=10, h_min=-5, h_max=16)

    def forward_impl(
        self,
        x: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        b, n, _, _, _ = x["imgs_qry"].shape
        imgs_query = rearrange(x["imgs_qry"], "b n c h w -> (b n) c h w")
        ground_features = self.ground_encoder(imgs_query)  # (B, N, C, H, W)
        ground_features = rearrange(
            ground_features, "(b n) c h w -> b n c h w", b=b, n=n
        )

        # Process Perspective View to Bird's Eye View
        feat_qry = self.PVToBEV(
            ground_features,
            x["extr"],
            x["intr"],
            x["res"],
            x["gsd"],
        )

        return {"feat_qry": feat_qry, "gsd_qry": x["gsd"]}  # no change in gsd
