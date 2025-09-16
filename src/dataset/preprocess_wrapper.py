import math
import random
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils.gps import latlon_to_delta_xy, meters_per_pixel
from utils.vehicle_pose import VehiclePose


class PreprocessWrapper(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        max_shift_m: float = 20.0,
        max_rot_deg: float = 10.0,
        ref_size: int = 512,
        qry_size: Tuple[int, int] = (240, 320),
        seed=None,
    ):
        self.dataset = dataset
        self.max_shift_m = max_shift_m
        self.max_rot_deg = max_rot_deg
        self.ref_size = ref_size
        self.qry_size = qry_size
        self._rng = random.Random(seed) if seed is not None else random

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        img_ref, transform_gt = self._process_reference(
            item["img_ref"],
            item["gps_ref"],
            item["gps_qry"],
            item["heading"],
            item["shift"],
            item["rot"],
        )

        # Resize query images
        img_qry = [img.resize(self.qry_size, Image.BILINEAR) for img in item["img_qry"]]

        # To tensor and normalize
        img_qry = [self.transform(img) for img in img_qry]
        img_qry = torch.stack(img_qry)

        return {
            "input": {
                "ref": self.transform(img_ref),  # To tensor and normalize
                "qry": img_qry,
                "matcher": transform_gt.heading,  # Supervise during training
            },
            "label": transform_gt,
        }

    def _process_reference(
        self,
        img_ref: Image.Image,
        gps_ref: Tuple[float, float],
        gps_qry: Tuple[float, float],
        heading: float,
        shift: Tuple[float, float] | None,
        rot: float | None,
    ):
        """
        Process the reference image by applying shift and rotation such that
        the augmented vehicle ends up at the image center.

        Returns
        -------
        img_ref_proc : PIL.Image.Image
            Processed reference image (same size as input).
        transform_gt : Tuple[float, float, float]
            Ground truth transformation (dx [m], dy [m], dtheta [deg]).
            ENU convention: x=east, y=north, positive rotation CCW.
        """
        dx_m, dy_m = latlon_to_delta_xy(*gps_ref, *gps_qry)  # E, N in meters

        mpp = meters_per_pixel(gps_ref[0], 18, 2)

        shift_rx = shift[0] if shift is not None else self._rng.uniform(-1.0, 1.0)
        shift_ry = shift[1] if shift is not None else self._rng.uniform(-1.0, 1.0)
        rot_val = rot if rot is not None else self._rng.uniform(-1.0, 1.0)

        aug_x_m = shift_rx * self.max_shift_m
        aug_y_m = shift_ry * self.max_shift_m

        aug_heading_deg = rot_val * self.max_rot_deg  # rotation augmentation (deg)

        dx_aug_m = dx_m + aug_x_m
        dy_aug_m = dy_m + aug_y_m
        dx_px = dx_aug_m / mpp  # east → +u (right)
        dy_px = -dy_aug_m / mpp  # north → -v (up), because image v grows downward

        W, H = img_ref.size
        cx_in = (W - 1) / 2.0
        cy_in = (H - 1) / 2.0

        theta = math.radians(heading + aug_heading_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        # PIL.Image.transform with AFFINE expects mapping from *output* -> *input*:
        # (x_in, y_in) = (a*x_out + b*y_out + c, d*x_out + e*y_out + f)
        # Build the inverse that yields: rotate around (cx_in,cy_in) then translate by
        # (dx_px, dy_px).
        a = cos_t
        b = -sin_t
        d = sin_t
        e = cos_t
        c = -a * cx_in - b * cy_in + cx_in + dx_px
        f = -d * cx_in - e * cy_in + cy_in + dy_px

        cx_out = (self.ref_size - 1) / 2.0
        cy_out = (self.ref_size - 1) / 2.0

        # Conceptually: we would first produce a W×H output where the vehicle is at
        # (cx_in, cy_in),
        # then crop a rectangle of size (out_W,out_H) whose centre is also
        # (cx_in,cy_in).
        # Cropping at top-left (x_tl, y_tl) = (cx_in - cx_out, cy_in - cy_out)
        # is equivalent to *pre*-shifting output coords by (x_tl, y_tl).
        x_tl = cx_in - cx_out
        y_tl = cy_in - cy_out

        # Substitute x_out := x_out + x_tl, y_out := y_out + y_tl in the inverse map:
        # c' = a*x_tl + b*y_tl + c ; f' = d*x_tl + e*y_tl + f
        c = a * x_tl + b * y_tl + c
        f = d * x_tl + e * y_tl + f

        coeffs = (a, b, c, d, e, f)

        img_ref_proc = img_ref.transform(
            (self.ref_size, self.ref_size),
            Image.AFFINE,
            coeffs,
            resample=Image.BILINEAR,
        )

        dx_px_true = dx_m / mpp
        dy_px_true = -dy_m / mpp

        # True vehicle position in input coords
        u_in_true = cx_in + dx_px_true
        v_in_true = cy_in + dy_px_true

        # Affine matrix
        A = np.array([[a, b], [d, e]])
        t = np.array([c, f])

        # Invert mapping: output = A^{-1} @ (input - t)
        A_inv = np.linalg.inv(A)
        pos_out = A_inv @ (np.array([u_in_true, v_in_true]) - t)

        u_true_out, v_true_out = pos_out

        # True heading relative to crop
        heading_true_out = -aug_heading_deg

        transform_gt = VehiclePose(
            uv=(u_true_out, v_true_out), heading=heading_true_out
        )

        return img_ref_proc, transform_gt
