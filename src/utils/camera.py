from abc import ABC, abstractmethod

import numpy as np


class Camera(ABC):
    """Abstract base class for camera calibration configs."""

    def __init__(self, cam_id: str):
        self.cam_id = cam_id

        extr, kind = self.load_extrinsics()

        if kind == "body2cam":
            self.body2cam = extr
            self.cam2body = np.linalg.inv(extr)
        elif kind == "cam2body":
            self.cam2body = extr
            self.body2cam = np.linalg.inv(extr)
        else:
            raise ValueError("kind must be 'body2cam' or 'cam2body'")

        self.body2ground = self.load_body2ground()  # 4x4
        self.K = self.load_intrinsics()  # 3x3
        self.width, self.height = self.load_resolution()  # int, int

    @abstractmethod
    def load_extrinsics(self) -> tuple[np.ndarray, str]:
        """Return (matrix, kind), where kind is 'body2cam' or 'cam2body'"""
        pass

    @abstractmethod
    def load_intrinsics(self) -> np.ndarray:
        """Return 3x3 intrinsic matrix."""
        pass

    @abstractmethod
    def load_resolution(self) -> tuple[int, int]:
        """Return (width, height)."""
        pass

    @abstractmethod
    def load_body2ground(self) -> np.ndarray:
        """Return 4x4 body to ground transformation matrix."""
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

    def get_cam2ground(self) -> np.ndarray:
        return self.body2ground @ self.cam2body

    def get_ground2cam(self):
        return np.linalg.inv(self.get_cam2ground())
