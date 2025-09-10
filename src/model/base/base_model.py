from typing import Dict

import torch
from torch import nn

from .base_encoder_qry import BaseEncoderQry
from .base_encoder_ref import BaseEncoderRef
from .base_matcher import BaseMatcher


class BaseModel(nn.Module):
    def __init__(
        self,
        ref_encoder: BaseEncoderRef,
        qry_encoder: BaseEncoderQry,
        matcher: BaseMatcher,
    ):
        super().__init__()

        # --- Type safety ---
        self._check_type(ref_encoder, BaseEncoderRef, "ref_encoder")
        self._check_type(qry_encoder, BaseEncoderQry, "qry_encoder")
        self._check_type(matcher, BaseMatcher, "matcher")

        # --- Submodules ---
        self.ref_encoder = ref_encoder
        self.qry_encoder = qry_encoder
        self.matcher = matcher

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            batch: dict mit mindestens
                - "input_ref": Tensor für Referenz
                - "input_qry": Tensor für Query
                - "input_matcher": optionale Zusatzinfos für Matcher

        Returns:
            Tensor: Matcher-Ausgabe (z.B. Score oder Volumen)
        """
        # 1. Extract Features
        ref_feat = self.ref_encoder(batch["input_ref"])  # [B, M, H, W]
        qry_feat = self.qry_encoder(batch["input_qry"])  # [B, M, h, w]

        # 2. Match Features
        out = self.matcher(
            ref_feat,
            qry_feat,
            batch.get(
                "input_matcher",
                None,
            ),
        )
        return out

    @staticmethod
    def _check_type(instance, base_class, name: str):
        if not isinstance(instance, base_class):
            raise TypeError(
                (
                    f"{name} must inherit from {base_class.__name__}, "
                    f"got {type(instance).__name__}"
                )
            )
