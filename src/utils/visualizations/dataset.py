import os
from typing import List, Tuple

import folium


def visualize_train_test(
    train_points: List[Tuple[float, float]],
    test_points: List[Tuple[float, float]],
    out_name: str,
    zoom_start: int = 17,
) -> str:
    """
    Create a folium map with train (red) and test (green) GPS points and save it to HTML.

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
