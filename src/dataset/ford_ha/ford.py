from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from utils.camera import Camera

from ..base_dataset import BaseDataset
from .config import (
    CAM_MAP,
    DATA_DIR,
    LOGS,
)


class FordCamera(Camera):
    """Camera config for the Ford dataset (V2 calibration)."""

    def __init__(self, cam_id: str, data_dir: Path):
        self.data_dir = data_dir
        super().__init__(cam_id)

    def load_extrinsics(self) -> np.ndarray:
        path = self.data_dir / f"camera{CAM_MAP[self.cam_id]}_body.yaml"
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        t = np.array(
            [
                data["transform"]["translation"]["x"],
                data["transform"]["translation"]["y"],
                data["transform"]["translation"]["z"],
            ]
        )
        q = np.array(
            [
                data["transform"]["rotation"]["x"],
                data["transform"]["rotation"]["y"],
                data["transform"]["rotation"]["z"],
                data["transform"]["rotation"]["w"],
            ]
        )

        R_mat = R.from_quat(q).as_matrix()

        T = np.eye(4)
        T[:3, :3] = R_mat
        T[:3, 3] = t
        return T, "cam2body"

    def load_body2ground(self):
        T_body2ground = np.eye(4)
        T_body2ground[2, 3] = -0.334585
        return T_body2ground

    def load_intrinsics(self) -> np.ndarray:
        path = self.data_dir / f"camera{CAM_MAP[self.cam_id]}Intrinsics.yaml"
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return np.array(data["K"]).reshape(3, 3)

    def load_resolution(self) -> tuple[int, int]:
        path = self.data_dir / f"camera{CAM_MAP[self.cam_id]}Intrinsics.yaml"
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return data["width"], data["height"]


class Ford(BaseDataset):
    """Dataset wrapper for the Ford HA dataset.

    Keeps this class minimal: it defers heavy lifting to `BaseDataset` and
    provides dataset-specific helpers for loading frames and camera configs.
    """

    def __init__(
        self,
        split: str,
        cams: List,
        log: int,
        sorted: bool = False,
        random_cams: bool = False,
        neg_samples: int = 0,
    ) -> None:
        # Initialize BaseDataset with dataset-specific dataframe and camera configs
        super().__init__(
            dataframe=self._load_dataframe(split, cams, log, sorted),
            cam_cfgs=self._load_cams(cams),
            random_cams=random_cams,
            neg_samples=neg_samples,
        )

    @staticmethod
    def _load_dataframe(
        split: str, cams: List[str], log: int, sorted: bool
    ) -> pd.DataFrame:
        """Load Ford dataset dataframe for a given split and log."""

        cfg = LOGS[split][log]
        log_path, indices = cfg["path"], cfg["indices"]

        # 1. Load raw info dataframe
        info_file = Ford._get_info_file(split, sorted)
        info_path = log_path / info_file
        df = Ford._read_info_file(info_path)

        # 2. Sort if required
        if sorted:
            df = Ford._sort_by_grdname(df)

        # 3. Filter by valid indices if split is train
        df = (
            df.iloc[indices].reset_index(drop=True)
            if split == "train" or sorted
            else df
        )

        # 4. Enrich with dataset-specific columns
        return Ford._build_dataframe(df, cams, log_path)

    # --- helpers ---

    @staticmethod
    def _get_info_file(split: str, sorted: bool) -> str:
        """Get the appropriate info file name based on the split."""
        if split == "test" and not sorted:
            return "grd_sat_quaternion_latlon_test.txt"
        else:
            return "grd_sat_quaternion_latlon.txt"

    @staticmethod
    def _read_info_file(info_path: Path) -> pd.DataFrame:
        """Read Ford info file and assign appropriate columns."""
        expected_cols = [
            "grd_name",
            "q0",
            "q1",
            "q2",
            "q3",
            "g_lat",
            "g_lon",
            "s_lat",
            "s_lon",
            "shift_u",
            "shift_v",
            "rot",
        ]

        dtype_map = {
            "grd_name": str,
            "q0": "float64",
            "q1": "float64",
            "q2": "float64",
            "q3": "float64",
            "g_lat": "float64",
            "g_lon": "float64",
            "s_lat": str,
            "s_lon": str,
            "shift_u": "float64",
            "shift_v": "float64",
            "rot": "float64",
        }

        df = pd.read_csv(info_path, sep=" ", header=None, dtype=dtype_map)
        df.columns = expected_cols[: df.shape[1]]
        return df

    @staticmethod
    def _sort_by_grdname(df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe numerically by grd_name field."""
        df = df.assign(
            grd_num=df["grd_name"].str.replace(".png", "", regex=False).astype(int)
        )
        return df.sort_values(by="grd_num", ascending=True).reset_index(drop=True)

    @staticmethod
    def _build_dataframe(
        df: pd.DataFrame, cams: List[str], log_path: Path
    ) -> pd.DataFrame:
        """Add dataset-specific columns (paths, gps, yaw, etc.)."""
        df = df.assign(
            img_ref_path=df.apply(
                lambda row: log_path
                / "SatelliteMaps_18"
                / f"{row['s_lat']}_{row['s_lon']}.png",
                axis=1,
            ),
            img_qry_path=df.apply(
                lambda row: {
                    cam: log_path
                    / f"{'-'.join(log_path.relative_to(DATA_DIR).parts)}-{cam}"
                    / row["grd_name"]
                    for cam in cams
                },
                axis=1,
            ),
            gps_ref=df.apply(
                lambda row: (float(row["s_lat"]), float(row["s_lon"])), axis=1
            ),
            gps_qry=df.apply(
                lambda row: (float(row["g_lat"]), float(row["g_lon"])), axis=1
            ),
            yaw=df.apply(
                lambda row: np.arctan2(
                    2.0 * (row["q3"] * row["q0"] + row["q1"] * row["q2"]),
                    -1.0 + 2.0 * (row["q0"] ** 2 + row["q1"] ** 2),
                )
                / np.pi
                * 180.0,
                axis=1,
            ),
        )

        # If both shift_u and shift_v exist, combine them into a single shift column
        if "shift_u" in df.columns and "shift_v" in df.columns:
            df = df.assign(
                shift=df.apply(lambda row: (row["shift_u"], row["shift_v"]), axis=1)
            )

        # Base required columns
        cols = ["img_ref_path", "img_qry_path", "gps_ref", "gps_qry", "yaw"]

        # Add optional columns if present
        if "shift" in df.columns:
            cols.append("shift")
        if "rot" in df.columns:
            cols.append("rot")

        return df[cols]

    @staticmethod
    def _load_cams(cams: List) -> Dict[str, Camera]:
        """Instantiate FordCamera objects for each cam."""
        assert all(cam in CAM_MAP for cam in cams), (
            "Some cameras not found in CAM_MAP. "
            f"Available cameras: {list(CAM_MAP.keys())}"
        )

        return {cam: FordCamera(cam, DATA_DIR / "Calibration" / "V2") for cam in cams}


if __name__ == "__main__":
    from utils.visualizations.dataset import visualize_train_test

    for sorted in [False, True]:
        for log in LOGS["train"].keys():
            train_points, test_points = [], []

            # Collect train points for this log
            df_train = Ford(
                split="train",
                cams=["FL"],
                log=log,
                sorted=sorted,
                random_cams=False,
            ).df_raw
            train_points.extend(df_train["gps_qry"].tolist())

            # Collect test points for this log
            df_test = Ford(
                split="test",
                cams=["FL"],
                log=log,
                sorted=sorted,
                random_cams=False,
            ).df_raw
            test_points.extend(df_test["gps_qry"].tolist())

            # Make separate visualization per log
            visualize_train_test(
                train_points=train_points,
                test_points=test_points,
                out_name=(
                    f"ford_ha_log_{log}.html"
                    if not sorted
                    else f"ford_ha_log_{log}_sorted.html"
                ),
                zoom_start=17,
            )
