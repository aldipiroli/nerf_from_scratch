import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from dataset.download_dataset import download_tiny_nerf_dataset


def visualize_camera_poses(camera_poses):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    ax.scatter(0, 0, 0, color="black", s=100)  # object

    for pose in camera_poses:
        x, y, z, _ = pose[:, -1]
        ax.scatter(x, y, z, color="blue")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


if __name__ == "__main__":
    data_path = download_tiny_nerf_dataset()
    data = np.load(data_path)
    visualize_camera_poses(data["poses"])
