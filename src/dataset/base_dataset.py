import random
from typing import Any, Dict, Hashable, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.Dataframe,
        cam_cfgs: Dict[str, Any],
        random_cams: bool = False,
        neg_samples: int = 0,
    ):
        super().__init__()

        self.random_cams = random_cams
        self.neg_samples = neg_samples

        self.df_raw = dataframe
        self.cam_cfgs = cam_cfgs

    def __len__(self) -> int:
        return len(self.df_epoch)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df_epoch.iloc[idx]

        # Loading reference image
        img_ref = Image.open(row["img_ref_path"]).convert("RGB")

        # Loading query images and camera configurations
        cams = self._get_cams()
        img_qry = [Image.open(row["img_qry_path"][k]).convert("RGB") for k in cams]
        cam_cfgs = {k: [self.cam_cfgs[k]] for k in cams}

        return {
            "img_ref": img_ref,
            "img_qry": img_qry,
            "cam_cfgs": cam_cfgs,
            "gps_ref": row["gps_ref"],  # GPS of ref image
            "gps_qry": row["gps_qry"],  # GPS of vehicle
            "bearing": row["bearing"],  # bearing wrt. North
            "shift": row.get("shift"),  # None if not present
            "rot": row.get("rot"),  # None if not present
            "neg_samples": self._load_neg_samples(),
        }

    def _load_image(self, path: str) -> Any:
        return Image.open(path).convert("RGB")

    def _get_cams(self) -> List[Hashable]:
        all_keys = list(self.cam_cfgs.keys())

        if not self.random_cams:
            return all_keys

        n = random.randint(1, len(all_keys))
        return random.sample(all_keys, n)

    def _load_neg_samples(self):
        """
        Load negative samples for a positive sample.

        Input:
            self (Dataset): dataset instance with attribute `neg_samples`.

        Output (dict):
            {
                "neg_img_ref": List[PIL.Image.Image],
                "neg_bearing": [float],
            }
            or None if no negative samples are loaded.
        """
        if self.neg_samples > 0:
            raise NotImplementedError("Negative samples not implemented yet")
        return None

    def resample(self, seed: int | None = None) -> None:
        """
        Resample dataset at the beginning of an epoch.
        Base implementation: simply copies df_raw into df_epoch.

        Args:
            seed (int | None): optional random seed (can be used in subclasses).
        """
        self.df_epoch = self.df_raw
