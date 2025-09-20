import math
import random
from dataclasses import dataclass
from typing import Tuple, cast

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.base_dataset import BaseDataset
from utils.gps import latlon_to_delta_xy, meters_per_pixel


@dataclass
class SampleInput:
    ref: torch.Tensor
    qry: torch.Tensor
    cams: tuple[str]
    matcher: torch.Tensor


@dataclass
class Sample:
    input: SampleInput
    label: torch.Tensor


class PreprocessWrapper(Dataset):
    def __init__(
        self,
        dataset: BaseDataset,
        max_shift_m: float = 20.0,
        max_rot_deg: float = 10.0,
        ref_size: int = 512,
        qry_scale: float = 0.25,
        grid_size: Tuple[int, int] = (40, 40),
        grid_upsampling_fac: float = 8,
        seed=None,
    ):
        self.dataset = dataset
        self.max_shift_m = max_shift_m
        self.max_rot_deg = max_rot_deg
        self.ref_size = ref_size
        self.qry_scale = qry_scale
        self.grid_size = grid_size
        self.grid_upsampling_fac = grid_upsampling_fac

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
        return len(self.dataset.df_epoch)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]

        img_ref, uv, heading = self._process_reference(
            item.img_ref,
            item.gps_ref,
            item.gps_qry,
            item.heading,
            item.shift,
            item.rot,
        )
        print(item.cams)
        return Sample(
            input=SampleInput(
                ref=img_ref,
                qry=self._process_query_images(item.img_qry),
                cams=item.cams,  # einfach Liste von Strings
                matcher=torch.tensor([heading], dtype=torch.float32),
            ),
            label=torch.tensor(uv + [heading], dtype=torch.float32),
        )

    def _process_reference(
        self,
        img_ref: Image.Image,
        gps_ref: Tuple[float, float],
        gps_qry: Tuple[float, float],
        heading: float,
        shift: Tuple[float, float] | None,
        rot: float | None,
    ) -> tuple[torch.Tensor, list[float], float]:
        """
        Align and augment a reference image based on GPS and heading information.
        Returns the transformed image, true vehicle position, and heading.
        """
        # --- step 1: GPS-based delta + augmentations ---
        dx_px, dy_px, aug_heading_deg, mpp = self._compute_augmented_shift(
            gps_ref, gps_qry, shift, rot
        )

        # --- step 2: affine transform coefficients ---
        coeffs = self._build_affine(
            img_ref.size, dx_px, dy_px, heading + aug_heading_deg
        )

        # --- step 3: apply transform & normalize ---
        img_ref_proc = img_ref.transform(
            (self.ref_size, self.ref_size),
            Image.Transform.AFFINE,
            coeffs,
            resample=Image.Resampling.BICUBIC,
        )
        # img_ref_proc = self.transform(img_ref_proc)
        img_ref_proc = cast(torch.Tensor, self.transform(img_ref_proc))
        # --- step 4: compute ground-truth position & heading ---
        uv, heading_true_out = self._compute_ground_truth(
            img_ref.size, gps_ref, gps_qry, coeffs, mpp, aug_heading_deg
        )

        return img_ref_proc, uv, heading_true_out

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _compute_augmented_shift(
        self,
        gps_ref: Tuple[float, float],
        gps_qry: Tuple[float, float],
        shift: Tuple[float, float] | None,
        rot: float | None,
    ) -> tuple[float, float, float, float]:
        dx_m, dy_m = latlon_to_delta_xy(*gps_ref, *gps_qry)
        mpp = meters_per_pixel(gps_ref[0], 18, 2)

        shift_rx = shift[0] if shift is not None else self._rng.uniform(-1.0, 1.0)
        shift_ry = shift[1] if shift is not None else self._rng.uniform(-1.0, 1.0)
        rot_val = rot if rot is not None else self._rng.uniform(-1.0, 1.0)

        dx_aug_m = dx_m + shift_rx * self.max_shift_m
        dy_aug_m = dy_m + shift_ry * self.max_shift_m
        aug_heading_deg = rot_val * self.max_rot_deg

        dx_px = dx_aug_m / mpp
        dy_px = -dy_aug_m / mpp
        return dx_px, dy_px, aug_heading_deg, mpp

    def _build_affine(
        self, size: tuple[int, int], dx_px: float, dy_px: float, heading: float
    ) -> tuple[float, float, float, float, float, float]:
        W, H = size
        cx_in = (W - 1) / 2.0
        cy_in = (H - 1) / 2.0

        theta = math.radians(heading)
        cos_t, sin_t = math.cos(theta), math.sin(theta)

        a, b, d, e = cos_t, -sin_t, sin_t, cos_t
        c = -a * cx_in - b * cy_in + cx_in + dx_px
        f = -d * cx_in - e * cy_in + cy_in + dy_px

        cx_out = (self.ref_size - 1) / 2.0
        cy_out = (self.ref_size - 1) / 2.0
        x_tl, y_tl = cx_in - cx_out, cy_in - cy_out

        c += a * x_tl + b * y_tl
        f += d * x_tl + e * y_tl

        return a, b, c, d, e, f

    def _compute_ground_truth(
        self,
        size: tuple[int, int],
        gps_ref: Tuple[float, float],
        gps_qry: Tuple[float, float],
        coeffs: tuple[float, float, float, float, float, float],
        mpp: float,
        aug_heading_deg: float,
    ) -> tuple[list[float], float]:
        dx_m, dy_m = latlon_to_delta_xy(*gps_ref, *gps_qry)
        dx_px_true = dx_m / mpp
        dy_px_true = -dy_m / mpp

        W, H = size
        cx_in, cy_in = (W - 1) / 2.0, (H - 1) / 2.0
        u_in_true, v_in_true = cx_in + dx_px_true, cy_in + dy_px_true

        a, b, c, d, e, f = coeffs
        A = np.array([[a, b], [d, e]])
        t = np.array([c, f])

        A_inv = np.linalg.inv(A)
        u_true_out, v_true_out = A_inv @ (np.array([u_in_true, v_in_true]) - t)

        heading_true_out = -aug_heading_deg
        return [u_true_out, v_true_out], heading_true_out

    def _process_query_images(self, imgs: list[Image.Image]) -> torch.Tensor:
        """
        Resize, normalize, and stack query images.

        Args:
            imgs: List of PIL images.

        Returns:
            Tensor [N, C, H, W] of processed query images.
        """
        imgs = [
            img.resize(
                (
                    max(1, int(img.width * self.qry_scale)),
                    max(1, int(img.height * self.qry_scale)),
                ),
                Image.Resampling.LANCZOS,
            )
            for img in imgs
        ]
        tensor_imgs = [cast(torch.Tensor, self.transform(img)) for img in imgs]
        return torch.stack(tensor_imgs)

    # TODO: negative samples
    def _process_neg_samples(self, neg_samples):
        if neg_samples is None:
            return None
        else:
            raise NotImplementedError("Negative samples processing not implemented yet")
