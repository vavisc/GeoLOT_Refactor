import math
import os
from typing import List, Tuple

import folium
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from utils.vehicle_pose import VehiclePose


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
