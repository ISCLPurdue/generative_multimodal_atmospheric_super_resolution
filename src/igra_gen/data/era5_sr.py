import os
from glob import glob
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange

class ERA5SR(Dataset):
    def __init__(
        self,
        root: str,
        variables: list[str],
        conditions: int = 1, # last ones are the conditional variables
        split: str = "train",
        rescale = 1,
    ):
        super().__init__()
        self.root = root
        self.files = sorted(glob(os.path.join(root, split, "*.h5")))
        self.variables = variables
        self.conditions = conditions
        means = np.load(os.path.join(root, f"normalize_mean.npz"))
        self.means = np.stack([means[v] for v in variables], axis=0).reshape(-1, 1, 1)
        stds = np.load(os.path.join(root, f"normalize_std.npz"))
        self.stds = np.stack([stds[v] for v in variables], axis=0).reshape(-1, 1, 1)

        self.shape = self._load_file(
            self.files[np.random.randint(0, len(self.files))], variables
        ).shape
        self.scale = rescale

    @property
    def n_channels(self):
        assert len(self.shape) == 3
        return self.shape[0]*self.scale-self.conditions
    
    @property
    def cond_channels(self):
        return self.conditions

    @property
    def img_resolution(self):
        return self.shape[1], self.shape[2]

    def get_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        lat = np.load(os.path.join(self.root, "lat.npy")).astype(np.float32)
        lon = np.load(os.path.join(self.root, "lon.npy")).astype(np.float32)
        return lat, lon

    def get_time(self, idx: int) -> np.datetime64:
        with h5py.File(self.files[idx], "r") as f:
            timestamp = f["input"]["time"][()]
        return np.datetime64(timestamp.decode("utf-8"))

    def _load_file(self, path: str, variables: list[str]) -> np.ndarray:
        with h5py.File(path, "r") as f:
            data = {
                main_key: {
                    sub_key: np.array(value)
                    for sub_key, value in group.items()
                    if sub_key in variables + ["time"]
                }
                for main_key, group in f.items()
                if main_key in ["input"]
            }
        x = np.stack([data["input"][v] for v in variables], axis=0)
        return x

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        x =  (x - self.means) / self.stds
        return x

    def _unstandardize(self, x: np.ndarray) -> np.ndarray:
        # No surface geopotential in output
        x =  x * self.stds[:-1] + self.means[:-1]
        return x

    def __len__(self) -> int:
        return len(self.files)  # last files dont have a target

    def __getitem__(self, idx: int):
        t = torch.from_numpy(
            self._standardize(self._load_file(self.files[idx], self.variables)[:,:720,:1440])
        )
        x = t[-1:] 
        return x,t[:-1]  # C x H x W
