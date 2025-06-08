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


def get_orthogonal_vectors(v):
    if v[0] != 0 or v[1] != 0:
        u = np.array([0, 0, 1])
    else:
        u = np.array([0, 1, 0])
    orthogonal = np.cross(v, u)

    dot = np.dot(v, orthogonal)
    assert np.isclose(dot, 0)
    return orthogonal


def draw_plane(ax, v=(0, 0, 1), p=(0, 0, 0), h=1, w=1):
    v = np.array(v, dtype=float)
    p = np.array(p, dtype=float)
    v = v / np.linalg.norm(v)

    v1 = get_orthogonal_vectors(v)
    v2 = np.cross(v, v1)
    assert np.isclose(np.dot(v2, v), 0)

    i = np.linspace(-h / 2, h / 2, 10)
    j = np.linspace(-w / 2, w / 2, 10)
    ii, jj = np.meshgrid(i, j)
    points = ii[:, :, None] * v1[None, None, :] + jj[:, :, None] * v2[None, None, :] + p[None, None, :]

    x = points[:, :, 0]
    y = points[:, :, 1]
    z = points[:, :, 2]
    ax.plot_surface(x, y, z, alpha=0.5)
    return ax


def get_unit_vector(start, end):
    vec = end - start
    norm = np.linalg.norm(vec)
    vec_norm = vec / norm
    vec_unit = start + vec_norm
    return vec_unit


def draw_vector_start_end(ax, start, end, color="r", unit_vector=True):
    if unit_vector:
        end = get_unit_vector(start, end)

    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
    ax.plot(end[0], end[1], end[2], marker="D", color=color)
    return ax


def plot_point(ax, p, color="b"):
    ax.plot(p[0], p[1], p[2], marker="o", color=color)
    return ax


def get_vector_from_points(a, b, normalize=True):
    v = b - a
    return v


def vis_camera_poses(data):
    center = [0, 0, 0]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # object
    x, y, z = draw_sphere(center=(0, 0, 0), radius=0.1)
    ax.plot_surface(x, y, z)

    # cameras
    for cam in data:
        ax = plot_point(ax, p=(cam[:3, -1]))  # camera pos
        ax = draw_vector_start_end(ax, start=cam[:3, -1], end=center)  # principal dir
        v = get_unit_vector(start=cam[:3, -1], end=center)
        ax = draw_plane(ax, v, v)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    plt.show()


if __name__ == "__main__":
    data_path = download_tiny_nerf_dataset()
    data = np.load(data_path)
    vis_camera_poses(data["poses"][2:3])
