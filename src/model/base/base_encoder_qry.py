from typing import Dict

import torch
from torch import nn


class BaseEncoderQry(nn.Module):
    """
    Abstract base class for query encoders.

    Expected:
        Input:
            x: Dict[str, Tensor] with:
                - "imgs": Tensor [B, N, C, H, W]
                - "gsd": Tensor [B]
                - "intr": Tensor [B, 3, 3]
                - "extr": Tensor [B, 4, 4]
                - "res": Tensor [B, 2]
        Output:
            Dict[str, Tensor] with:
                - "feat_qry": Tensor [B, M, h, w]
                - "gsd_qry":  Tensor [B]
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # --- Call subclass implementation ---
        out = self.forward_impl(x)

        # --- Output checks ---
        if not isinstance(out, dict):
            raise TypeError("forward_impl must return a dict")

        if "feat_qry" not in out or "gsd_qry" not in out:
            raise KeyError("Output dict must contain keys 'feat_qry' and 'gsd_qry'")

        return out

    def forward_impl(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Must be implemented by subclasses.

        Args:
            x: Dict with:
                - "imgs": [B,N,C,H,W]
                - "gsd": [B]
                - "intr": [B,3,3]
                - "extr": [B,4,4]
                - "res": [B,2]
        Returns:
            Dict with:
                - "feat_qry": [B,M,h,w]
                - "gsd_qry": [B]
        """
        raise NotImplementedError
