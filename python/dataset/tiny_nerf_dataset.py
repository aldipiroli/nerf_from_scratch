from pathlib import Path

import numpy as np
import torch
from dataset.download_dataset import download_tiny_nerf_dataset
from torch.utils.data import Dataset
from utils.render import get_ray_vectors, sample_ray_vecotrs


class TinyNeRFDataset(Dataset):
    def __init__(self, root_dir, mode="train", N=10, M=20, tn=1, tf=3, H=100, W=100):
        self.root_dir = Path(root_dir)
        self.mode = mode
        download_tiny_nerf_dataset(data_dir=self.root_dir)
        data = np.load(self.root_dir / "tiny_nerf_data.npz")

        self.images = torch.tensor(data["images"])
        self.poses = torch.tensor(data["poses"])
        self.focal = torch.tensor(data["focal"])
        self.N_train = 100
        self.N = N
        self.M = M
        self.tn = tn
        self.tf = tf
        self.H = H
        self.W = W

        if mode == "train":
            self.images = self.images[: self.N_train]
            self.poses = self.poses[: self.N_train]
        else:
            self.images = self.images[self.N_train :]
            self.poses = self.poses[self.N_train :]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        dirs, origin = get_ray_vectors(self.poses[idx], self.focal, H=self.H, W=self.W)
        all_ray_queries, (samples_i, samples_j) = sample_ray_vecotrs(
            dirs, origin, N=self.N, M=self.M, tn=self.tn, tf=self.tf, H=self.H, W=self.W, mode=self.mode
        )
        image_gt_colors = self.images[idx][samples_i, samples_j]
        return all_ray_queries, image_gt_colors
