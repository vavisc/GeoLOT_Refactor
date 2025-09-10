from typing import Dict

import torch
from torch import nn


class BaseMatcher(nn.Module):
    """
    Abstract base class for query encoders.

    Expected input:
        Dict with:
            - "feat_ref": Tensor [B,C,H,W]
            - "gsd_ref": Tensor [B]
            - "feat_qry": Tensor [B,C,h,w]
            - "gsd_qry": Tensor [B]
            - "t_bp": Tensor [B,4,4] (optional, only for training) Transform body to
                prior
            - "feat_ref_neg": Tensor [B,A,H,W] (optional, only for training)

    Expected output:
        Dict with:
            - "t_bp_pred": Tensor [B,4] # Transform body to prior
            - "opt_target": Tensor [B,A,H,W]
            - "rot_feat_qry": Tensor [B,A,H,W] # Only to check if correctly rotated
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # --- Input shape check ---
        assert torch.allclose(x["gsd_ref"], x["gsd_qry"], rtol=0.0, atol=1e-4), (
            "GSD mismatch between ref and qry"
        )

        # --- Call subclass implementation ---
        out = self.forward_impl(x)

        # --- Output checks ---
        if not isinstance(out, dict):
            raise TypeError("forward_impl must return a dict")

        if (
            "transform" not in out
            or "opt_target" not in out
            or "rotated_feat_query" not in out
        ):
            raise KeyError(
                "Output dict must contain keys 'feat_qry', 'gsd_qry' and 'rot_feat_qry'"
            )

        return out

    def forward_impl(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Must be implemented by subclasses.

        Args:
            x: [B,C,H,W]

        Returns:
            y: [B,M,H,W]
        """
        raise NotImplementedError
