import math
import os
from typing import List, Tuple

import folium
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms


def visualize_train_test(
    train_points: List[Tuple[float, float]],
    test_points: List[Tuple[float, float]],
    out_name: str,
    zoom_start: int = 17,
) -> str:
    """
    Create a folium map with train (red) and test (green) GPS points and save it to
    HTML.

    Args:
        train_points: List of (lat, lon) tuples for train samples.
        test_points: List of (lat, lon) tuples for test samples.
        out_name: Name of the output HTML file.
        zoom_start: Initial zoom level of the map.

    Returns:
        Path to the saved HTML file.
    """
    if not train_points and not test_points:
        raise ValueError("No points provided for visualization.")

    # Pick first valid point as map center
    center = None
    for points in (train_points, test_points):
        if points:
            center = (float(points[0][0]), float(points[0][1]))
            break

    fmap = folium.Map(
        location=center,
        zoom_start=zoom_start,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )

    # Train = red
    for lat, lon in train_points:
        folium.CircleMarker(
            location=(float(lat), float(lon)),
            radius=2,
            color="#e41a1c",
            fill=True,
            fill_color="#e41a1c",
            fill_opacity=1.0,
            opacity=1.0,
        ).add_to(fmap)

    # Test = green
    for lat, lon in test_points:
        folium.CircleMarker(
            location=(float(lat), float(lon)),
            radius=2,
            color="#4daf4a",
            fill=True,
            fill_color="#4daf4a",
            fill_opacity=1.0,
            opacity=1.0,
        ).add_to(fmap)

    # Save under project root: outputs/datasets/visualizations
    src_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    repo_root = os.path.dirname(src_dir)
    out_dir = os.path.join(repo_root, "outputs", "datasets", "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, out_name)
    fmap.save(out_path)

    print(f"Saved visualization: {out_path}")
    return out_path


def unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Undo ImageNet normalization for a tensor in [C, H, W].
    Returns a tensor in [0, 1].
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean


def tensor_to_pil(tensor: torch.Tensor):
    """
    Unnormalize an ImageNet-normalized tensor and convert to PIL.Image.
    """
    tensor = unnormalize(tensor)  # back to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)  # ensure valid range
    return transforms.ToPILImage()(tensor.cpu())


def _draw_vehicle_pose(
    draw: ImageDraw.ImageDraw,
    pose: VehiclePose,
    *,
    color="orange",
    radius_px=10.0,
    line_len_px=24.0,
    linewidth=4,
):
    """Draw a vehicle pose marker on a PIL image."""
    if pose is None:
        return

    u, v = pose.uv
    theta = math.radians(pose.heading)
    dx = line_len_px * math.sin(theta)
    dy = -line_len_px * math.cos(theta)

    # Circle for vehicle body
    left_up = (u - radius_px, v - radius_px)
    right_down = (u + radius_px, v + radius_px)
    draw.ellipse([left_up, right_down], outline=color, width=linewidth)

    # Heading line
    draw.line([u, v, u + dx, v + dy], fill=color, width=linewidth)

    # Small dot in the center of circle
    dot_r = 2
    draw.ellipse([u - dot_r, v - dot_r, u + dot_r, v + dot_r], fill=color)


def draw_vehicle_on_image(
    img,
    gt_pose: VehiclePose | None = None,
    pred_pose: VehiclePose | None = None,
    *,
    mark_center: bool = False,
) -> Image.Image:
    """
    Draw vehicle poses directly onto an image (PIL or torch.Tensor).

    Parameters
    ----------
    img : PIL.Image or torch.Tensor
        Input image (if tensor, must be normalized like ImageNet).
    gt_pose : VehiclePose, optional
        Ground-truth vehicle pose (drawn in green).
    pred_pose : VehiclePose, optional
        Prediction vehicle pose (drawn in red).
    mark_center : bool
        Whether to mark the image center with a red dot.

    Returns
    -------
    PIL.Image
        Edited image with drawings.
    """
    # Convert tensor -> PIL if needed
    if isinstance(img, torch.Tensor):
        pil_img = tensor_to_pil(img)
    elif isinstance(img, Image.Image):
        pil_img = img.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    draw = ImageDraw.Draw(pil_img)
    W, H = pil_img.size

    if mark_center:
        dot_r = 3
        draw.ellipse(
            [W / 2 - dot_r, H / 2 - dot_r, W / 2 + dot_r, H / 2 + dot_r], fill="red"
        )

    if gt_pose is not None:
        _draw_vehicle_pose(draw, gt_pose, color="#39FF14")

    if pred_pose is not None:
        _draw_vehicle_pose(draw, pred_pose, color="red")

    return pil_img


def visualize_grid_on_image(
    image: torch.Tensor,  # [C, H, W] in [0,1] or [0,255]
    grid: torch.Tensor,  # [H_out, W_out, 2] in [-1,1]
    point_size: int = 3,
    gamma: float = 0.6,  # <1 = more saturated blend
) -> Image.Image:
    """Overlay sampled grid points on an image with smoothly interpolated, vibrant
    corner colors."""

    # --- prep image ---
    if image.ndim == 3:  # [C, H, W]
        image = image.permute(1, 2, 0)  # -> [H, W, C]
    img_np = image.detach().cpu().numpy()
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    else:
        img_np = img_np.astype(np.uint8)

    pil_img = Image.fromarray(img_np)
    H, W = pil_img.size[1], pil_img.size[0]

    # --- grid to pixel coords ---
    u = ((grid[..., 0] + 1) / 2.0) * (W - 1)  # x coords
    v = ((grid[..., 1] + 1) / 2.0) * (H - 1)  # y coords
    u = u.cpu().numpy()
    v = v.cpu().numpy()

    # --- vibrant corner colors (R,G,B) ---
    c_tl = np.array([255, 0, 128])  # magenta
    c_tr = np.array([0, 200, 255])  # cyan
    c_bl = np.array([255, 220, 0])  # yellow
    c_br = np.array([255, 80, 0])  # bright orange

    # --- bilinear interpolation of colors ---
    H_out, W_out = grid.shape[:2]
    yi = np.linspace(0, 1, H_out)[:, None, None]  # (H_out,1,1)
    xi = np.linspace(0, 1, W_out)[None, :, None]  # (1,W_out,1)

    # mix in linear space
    colors = (
        (1 - xi) * (1 - yi) * c_tl
        + xi * (1 - yi) * c_tr
        + (1 - xi) * yi * c_bl
        + xi * yi * c_br
    ) / 255.0  # normalize to [0,1]

    # apply gamma correction to boost saturation
    colors = np.clip(colors, 0, 1) ** gamma
    colors = (colors * 255).astype(np.uint8)

    # --- draw points ---
    draw = ImageDraw.Draw(pil_img)
    for i in range(H_out):
        for j in range(W_out):
            x, y = u[i, j], v[i, j]
            if 0 <= x < W and 0 <= y < H:
                color = tuple(colors[i, j])
                draw.ellipse(
                    [x - point_size, y - point_size, x + point_size, y + point_size],
                    fill=color,
                    outline=color,
                )

    return pil_img


def sample_image_with_grid(
    image: torch.Tensor,  # [C,H,W] in [0,1] or [0,255]
    grid: torch.Tensor,  # [H_out, W_out, 2] in [-1,1]
    mode: str = "bilinear",
    align_corners: bool = True,
) -> Image.Image:
    """Sample an image using a normalized grid and return as PIL.Image."""

    if image.ndim == 3:  # [C,H,W]
        image = image.unsqueeze(0)  # -> [1,C,H,W]

    if image.max() > 1.0:
        image = image / 255.0  # normalize to [0,1]

    # [H_out, W_out, 2] -> [1, H_out, W_out, 2]
    grid = grid.unsqueeze(0)

    # grid_sample: input [N,C,H,W], grid [N,H_out,W_out,2]
    sampled = F.grid_sample(image, grid, mode=mode, align_corners=align_corners)

    # -> [C,H_out,W_out]
    sampled = sampled.squeeze(0).detach().cpu()

    # convert to numpy
    sampled_np = (sampled.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    return Image.fromarray(sampled_np)
