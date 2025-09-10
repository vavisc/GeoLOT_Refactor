from typing import Dict

import torch
from torch import nn


class BaseEncoderRef(nn.Module):
    """
    Abstract base class for reference encoders.

    Expected input:
        Dict with:
            - "imgs_ref": Tensor [B,C,H,W]
            - "gsd_ref": Tensor [B]

    Expected output:
        Dict with:
            - "feat_ref": Tensor [B,M,h,w]
            - "gsd_ref": Tensor [B]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # --- Call subclass implementation ---
        out = self.forward_impl(x)

        # --- Output checks ---
        if not isinstance(out, dict):
            raise TypeError("forward_impl must return a dict")

        if "feat_ref" not in out or "gsd_ref" not in out:
            raise KeyError("Output dict must contain keys 'feat_ref' and 'gsd_ref'")
        return out

    def forward_impl(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Must be implemented by subclasses.
        """
        raise NotImplementedError
