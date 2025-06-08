from pathlib import Path

import numpy as np
from dataset.download_dataset import download_tiny_nerf_dataset
from torch.utils.data import Dataset
from utils.render import camera_ray_query


class TinyNeRFDataset(Dataset):
    def __init__(self, root_dir, split="train", N=10, M=20, tn=1, tf=3, img_plane_h=64, img_plane_w=64):
        self.root_dir = Path(root_dir)
        self.split = split
        download_tiny_nerf_dataset(data_dir=self.root_dir)
        data = np.load(self.root_dir / "tiny_nerf_data.npz")

        self.images = data["images"]
        self.poses = data["poses"]
        self.N_train = 100
        self.N = N
        self.M = M
        self.tn = tn
        self.tf = tf
        self.img_plane_h = img_plane_h
        self.img_plane_w = img_plane_w

        if split == "train":
            self.images = self.images[: self.N_train]
            self.poses = self.poses[: self.N_train]
        else:
            self.images = self.images[self.N_train :]
            self.poses = self.poses[self.N_train :]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        query_points = camera_ray_query(self.poses[idx], N=self.N, M=self.M, tn=self.tn, tf=self.tf)
        return self.images[idx], self.poses[idx], query_points
