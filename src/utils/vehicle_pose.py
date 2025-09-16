from dataclasses import dataclass

import torch


@dataclass
class VehiclePose:
    uv: tuple[float, float]
    heading: float

    def set_from_image(self, uv: tuple[float, float], heading: float) -> None:
        """Update the vehicle pose."""
        self.uv = uv
        self.heading = heading

    def as_tensor(self) -> torch.Tensor:
        """Return pose as torch tensor [x, y, heading]."""
        return torch.tensor([self.uv[0], self.uv[1], self.heading], dtype=torch.float32)
