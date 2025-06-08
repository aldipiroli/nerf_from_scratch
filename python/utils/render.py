import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


def get_sphere_points(center=(0, 0, 0), radius=1, n=20):
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    gamma = np.linspace(0, 2 * np.pi, n)
    theta = np.linspace(0, np.pi, n)
    gamma_m, theta_m = np.meshgrid(gamma, theta)

    x = radius * np.sin(theta_m) * np.cos(gamma_m) + center[0]
    y = radius * np.sin(theta_m) * np.sin(gamma_m) + center[1]
    z = radius * np.cos(theta_m) + center[2]
    return x, y, z


def draw_surface(ax, x, y, z):
    ax.plot_surface(x, y, z)


def get_orthogonal_vector(v):
    v = v.to(torch.float32)  # Ensure the input is a float tensor
    if v[0] != 0 or v[1] != 0:
        u = torch.tensor([0, 0, 1], dtype=torch.float32)
    else:
        u = torch.tensor([0, 1, 0], dtype=torch.float32)

    orthogonal = torch.linalg.cross(v, u)
    dot = torch.dot(v, orthogonal)
    assert torch.isclose(dot, torch.tensor(0.0, dtype=torch.float32))
    return orthogonal


def get_plane_points(v=(0, 0, 1), p=(0, 0, 0), h=1, w=1):
    v = torch.tensor(v, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)
    v = v / torch.norm(v)

    v1 = get_orthogonal_vector(v)
    v2 = torch.linalg.cross(v, v1)

    i = torch.linspace(-h / 2, h / 2, h)
    j = torch.linspace(-w / 2, w / 2, w)
    ii, jj = torch.meshgrid(i, j, indexing="ij")

    points = ii[:, :, None] * v1[None, None, :] + jj[:, :, None] * v2[None, None, :] + p[None, None, :]

    x = points[:, :, 0]
    y = points[:, :, 1]
    z = points[:, :, 2]
    return x, y, z


def get_unit_vector(start, end):
    start = start.float()
    end = end.float()

    vec = end - start
    norm = torch.norm(vec)
    vec_norm = vec / norm
    vec_unit = start + vec_norm
    return vec_unit


def draw_vector_start_end(ax, start, end, color="r", unit_vector=True):
    if unit_vector:
        end = get_unit_vector(start, end)

    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
    ax.plot(end[0], end[1], end[2], marker="D", color=color)


def plot_point(ax, p, color="b"):
    ax.plot(p[0], p[1], p[2], marker="o", color=color)


def get_vector_from_points(a, b, normalize=False):
    v = b - a
    if normalize:
        norm = np.linalg.norm(v)
        v /= norm
    return v


def draw_parametric_vector(ax, start, v, t, color="g"):
    end = start + t * v
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, alpha=0.5)
    ax.plot(end[0], end[1], end[2], marker="D", color="r", alpha=0.3)


def resize_image(image, target_h, target_w):
    image_ = image.unsqueeze(0).float()
    image_ = image_.permute(0, 3, 1, 2)
    image_resize = F.interpolate(image_, size=(target_h, target_w), mode="bilinear")
    image_resize = image_resize.squeeze(0).permute(1, 2, 0)
    return image_resize


def get_rays_camera_to_plane(image, cam_center, plane, tn=1, tf=3, M=10, N=20, img_plane_h=100, img_plane_w=100):
    # Sample N plane points
    x, y, z = plane
    all_plane_points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (n_points, 3) plane points
    random_indices = torch.randperm(all_plane_points.shape[0])[:N]
    plane_points = all_plane_points[random_indices]  # (N, 3)

    image_resize = resize_image(image, target_h=img_plane_h, target_w=img_plane_w)
    image_resize = image_resize.reshape(-1, 3)
    assert image_resize.shape == all_plane_points.shape
    image_gt_colors = image_resize[random_indices]

    # Query points
    all_ray_queries = []
    t_step = (tf - tn) / M
    for p in plane_points:
        for t in torch.arange(tn, tf, t_step):
            v = get_vector_from_points(cam_center, p)
            query_point = p + (t * v)
            all_ray_queries.append(torch.cat((query_point, v), dim=0))

    all_ray_queries = torch.stack(all_ray_queries).reshape(N, M, 6)  # (N, M, 6) 6: x, y, z, v1, v2, v3
    return all_ray_queries, image_gt_colors


def camera_ray_query(cam_pose, image, N=10, M=20, tn=1, tf=3, img_plane_h=100, img_plane_w=100):
    center = torch.tensor([0, 0, 0])
    cam_center = torch.tensor(cam_pose[:3, -1])
    v = get_unit_vector(start=cam_center, end=center)
    x, y, z = get_plane_points(v=v, p=v, h=img_plane_h, w=img_plane_w)  # image plane
    query_points, image_gt_colors = get_rays_camera_to_plane(
        image=image,
        cam_center=cam_center,
        plane=(x, y, z),
        tf=tf,
        tn=tn,
        N=N,
        M=M,
        img_plane_h=img_plane_h,
        img_plane_w=img_plane_w,
    )
    query_points = query_points.float()
    image_gt_colors = image_gt_colors.float()
    return query_points, image_gt_colors


def plot_camera_ray_query(cam_pose):
    center = np.array([0, 0, 0])
    batch_id = 0
    cam_center = np.array(cam_pose[batch_id, :3, -1])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # object
    x, y, z = get_sphere_points(center=center, radius=1)
    draw_surface(ax, x, y, z)

    plot_point(ax, p=cam_center)  # camera pos
    draw_vector_start_end(ax, start=cam_center, end=center)  # principal dir
    v = get_unit_vector(start=cam_center, end=center)
    x, y, z = get_plane_points(v=v, p=v)  # image plane
    # draw_surface(ax, x, y, z)
    query_points = get_rays_camera_to_plane(ax, cam_center, plane=(x, y, z))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    plt.show()


def get_transmittance(step, density, target_step):
    sum_val = torch.sum(density[:, :, :target_step] * step, dim=2)
    trans = torch.exp(-sum_val)
    return trans


def get_volume_rendering(color, density, step=0.1):
    # Eq. 3 https://arxiv.org/pdf/2003.08934
    b, N, M, _ = color.shape
    predicted_ray_colors = []
    for i in range(M):
        alpha_i = 1 - torch.exp(-density[:, :, i] * step)
        trans_i = get_transmittance(step, density, target_step=i)
        ray_color_i = trans_i * alpha_i * color[:, :, i]
        predicted_ray_colors.append(ray_color_i)

    predicted_ray_colors = torch.stack(predicted_ray_colors)
    predicted_ray_colors = torch.sum(predicted_ray_colors, dim=0)
    return predicted_ray_colors


if __name__ == "__main__":
    data_path = download_tiny_nerf_dataset()
    data = np.load(data_path)
    camera_ray_query(data["poses"][2:3])
