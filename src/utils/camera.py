from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    """Abstract base class for camera calibration configs."""

    def __init__(self, cam_id: str):
        self.cam_id = cam_id
        self.body2cam = self.load_body2cam()  # 4x4
        self.K = self.load_intrinsics()  # 3x3
        self.width, self.height = self.load_resolution()  # int, int

        self.cam2body = np.linalg.inv(self.body2cam)

    @abstractmethod
    def load_body2cam(self) -> np.ndarray:
        """Return 4x4 homogeneous transform from body to cam."""
        pass

    @abstractmethod
    def load_intrinsics(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix."""
        pass

    @abstractmethod
    def load_resolution(self) -> tuple[int, int]:
        """Return (width, height)."""
        pass

    # ---- convenience methods ----
    def transform_point(self, T: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Apply homogeneous transform T to 3D point p."""
        p_h = np.ones(4)
        p_h[:3] = p
        return (T @ p_h)[:3]

    def get_cam2body(self) -> np.ndarray:
        return self.cam2body

    def get_body2cam(self) -> np.ndarray:
        return self.body2cam

    def get_intrinsics(self) -> np.ndarray:
        return self.K

    def get_resolution(self) -> tuple[int, int]:
        return self.width, self.height
