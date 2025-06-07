import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
from dataset.download_dataset import download_tiny_nerf_dataset


def cartesian_to_polar(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    gamma = np.arccos(y / r)
    theta = np.arctan2(z, y)
    return r, gamma, theta


def visualize_camera_poses(camera_poses):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect([1, 1, 1])

    ax.scatter(0, 0, 0, color="black", s=100)  # object

    for pose in camera_poses:
        x, y, z, _ = pose[:, -1]
        ax.scatter(x, y, z, color="blue")
        ax.quiver(x, y, z, -x, -y, -z, length=0.3, normalize=True, color="red")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def draw_sphere(center=(0, 0, 0), radius=1, n=20):
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    gamma = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(0, np.pi, n)
    gamma_m, theta_m = np.meshgrid(gamma, theta)

    x = radius * np.sin(theta_m) * np.cos(gamma_m) + center[0]
    y = radius * np.sin(theta_m) * np.sin(gamma_m) + center[1]
    z = radius * np.cos(theta_m) + center[2]
    return x, y, z


def draw_plane(v=(0, 0, 1), p=(0, 0, 1)):
    # https://mathworld.wolfram.com/Plane.html
    a, b, c = v[0], v[1], v[2]
    d = a * p[0] + b * p[1] + c * p[2]

    x = np.linspace(-2, 2, 10)
    y = np.linspace(-2, 2, 10)
    x, y = np.meshgrid(x, y)
    z = 1 / c * (d - a * x - b * y)

    return x, y, z


def draw_vector(ax, start, end, color="r"):
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
    ax.plot(end[0], end[1], end[2], marker="D", color=color)
    return ax


def visualize_plane():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # x, y, z = draw_plane()
    # ax.plot_surface(x, y, z)

    ax = draw_vector(ax=ax, start=(1.5, 0, 0), end=(2, 0, 0))
    x, y, z = draw_sphere()
    ax.plot_surface(x, y, z)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    data_path = download_tiny_nerf_dataset()
    data = np.load(data_path)
    visualize_plane()
