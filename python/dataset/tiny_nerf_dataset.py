from pathlib import Path

import numpy as np
from dataset.download_dataset import download_tiny_nerf_dataset
from torch.utils.data import Dataset


class TinyNeRFDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split
        download_tiny_nerf_dataset(data_dir=self.root_dir)
        data = np.load(self.root_dir / "tiny_nerf_data.npz")

        self.images = data["images"]
        self.poses = data["poses"]
        self.N_train = 100

        if split == "train":
            self.images = self.images[: self.N_train]
            self.poses = self.images[: self.N_train]
        else:
            self.images = self.images[self.N_train :]
            self.poses = self.images[self.N_train :]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.poses[idx]
