from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


def build_grid(
    grid_size: Tuple[int, int],  # (H, W) in pixels
    grid_mpp: float,
    height: float,
) -> np.ndarray:
    H, W = grid_size

    xs = np.arange(H, dtype=np.float32)
    ys = np.arange(W, dtype=np.float32)

    xs = -(xs + 0.5 - H / 2.0) * grid_mpp
    ys = (ys + 0.5 - W / 2.0) * grid_mpp

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = np.full_like(X, height, dtype=np.float32)

    return np.stack([X, Y, Z], axis=-1)  # [H, W, 3]


def process_masks(
    uv_cam_dict: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Apply first-come-first-served rule to camera masks (torch version)."""
    # initialize occupancy mask from first entry
    first_mask = next(iter(uv_cam_dict.values()))["mask"]
    occupancy_mask = torch.zeros_like(first_mask, dtype=torch.bool)

    for cam, data in uv_cam_dict.items():
        uv_grid = data["uv_grid"]
        mask = data["mask"].bool()

        new_mask = mask & (~occupancy_mask)
        occupancy_mask = occupancy_mask | new_mask

        # write back in dict format
        uv_cam_dict[cam] = {
            "uv_grid": uv_grid,
            "mask": new_mask,
        }

    return uv_cam_dict


def fuse_uv_grids(
    uv_cam_dict: dict[str, dict[str, torch.Tensor]],
    width: int,
    height: int,
    pad: int = 1,
) -> torch.Tensor:
    """
    Fuse per-camera uv_grids into a single batched uv_grid [B, H, W, 2].
    Invalid positions are filled with (-10, -10).
    """
    first_uv = next(iter(uv_cam_dict.values()))["uv_grid"]
    B, H, W, _ = first_uv.shape

    # init fused uv_grid with sentinel value
    fused_uv = torch.full(
        (B, H, W, 2), -10.0, dtype=torch.float32, device=first_uv.device
    )

    occupancy_mask = torch.zeros((B, H, W), dtype=torch.bool, device=first_uv.device)

    for i, (cam, data) in enumerate(uv_cam_dict.items()):
        print(f"Processing cam {i}: {cam}")

        uv_grid = data["uv_grid"].clone()  # [B, H, W, 2]
        mask = data["mask"].bool()  # [B, H, W]

        # first-come-first-served
        new_mask = mask & (~occupancy_mask)
        occupancy_mask |= new_mask

        # scale uv_grid to pixel space and shift per camera
        uv_grid[..., 0] = uv_grid[..., 0] * width
        uv_grid[..., 1] = uv_grid[..., 1] * height + i * (height + pad)

        # fill fused_uv at valid positions
        fused_uv[new_mask] = uv_grid[new_mask]

    # number of cams
    num_cams = len(uv_cam_dict)
    print(f"Fused {num_cams} cameras into uv grid of shape {fused_uv.shape}")

    # normalize to [-1, 1] for grid_sample
    fused_uv[..., 0] = 2 * (fused_uv[..., 0] / width - 0.5)
    fused_uv[..., 1] = 2 * (fused_uv[..., 1] / ((height + pad) * num_cams) - 0.5)

    return fused_uv


def concat_imgs(
    imgs: torch.Tensor,  # [B, N, C, H, W]
    pad: int = 0,
) -> torch.Tensor:
    """Concatenate images from multiple cameras into a single tall image.

    Args:
        imgs (torch.Tensor): Input images of shape [B, N, C, H, W].
        pad (int): Number of pixels to pad between images.
    Returns:
        torch.Tensor: Concatenated image of shape [B, C, H*N + pad*(N-1), W].
    """
    B, N, C, H, W = imgs.shape

    # [B, N, C, H, W] -> [B*N, C, H, W]
    # imgs = rearrange(imgs, "b n c h w -> (b n) c h w")

    if pad > 0:
        # pad only in vertical direction (dim=-2 = height)
        imgs = F.pad(imgs, (0, 0, 0, pad))  # (left, right, top, bottom)

        # reshape back: now height is H+pad
        # imgs = rearrange(imgs, "(b n) c h w -> b n c h w", b=B, n=N)

        # drop last pad after final camera
        # imgs[:, -1, :, -pad:, :] = 0
    # else:
    # imgs = rearrange(imgs, "(b n) c h w -> b n c h w", b=B, n=N)

    # stack vertically
    out = rearrange(imgs, "b n c h w -> b c (n h) w")

    return out
