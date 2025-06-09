import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset.download_dataset import download_tiny_nerf_dataset


def get_sphere_points(center=(0, 0, 0), radius=1, n=20):
    # https://en.wikipedia.org/wiki/Spherical_coordinate_system
    gamma = torch.linspace(0, 2 * torch.pi, n)
    theta = torch.linspace(0, torch.pi, n)
    gamma_m, theta_m = torch.meshgrid(gamma, theta)

    x = radius * torch.sin(theta_m) * torch.cos(gamma_m) + center[0]
    y = radius * torch.sin(theta_m) * torch.sin(gamma_m) + center[1]
    z = radius * torch.cos(theta_m) + center[2]
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


def draw_parametric_vector(ax, start, v, t, color="lightgray"):
    end = start + t * v
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color, alpha=0.5)
    # ax.plot(end[0], end[1], end[2], marker="o", color="r", alpha=0.3)


def resize_image(image, target_h, target_w):
    image_ = image.unsqueeze(0).float()
    image_ = image_.permute(0, 3, 1, 2)
    image_resize = F.interpolate(image_, size=(target_h, target_w), mode="bilinear")
    image_resize = image_resize.squeeze(0).permute(1, 2, 0)
    return image_resize


def get_rays_camera_to_plane(image, cam_center, plane, tn=1, tf=3, M=10, N=20, H=100, W=100, mode="train"):
    # Sample N plane points
    x, y, z = plane
    all_plane_points = torch.stack((x, y, z), dim=-1).reshape(-1, 3)  # (n_points, 3) plane points
    random_indices = torch.randperm(all_plane_points.shape[0])[:N]
    if mode == "train":
        plane_points = all_plane_points[random_indices]  # (N, 3)
    else:
        plane_points = all_plane_points[:N]

    image_resize = resize_image(image, target_h=H, target_w=W)
    image_resize = image_resize.reshape(-1, 3)
    assert image_resize.shape == all_plane_points.shape
    if mode == "train":
        image_gt_colors = image_resize[random_indices]
    else:
        image_gt_colors = image_resize[:N]

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


def camera_ray_query_plane_to_image(cam_pose, image, N=10, M=20, tn=1, tf=3, H=100, W=100, mode="train"):
    center = torch.tensor([0, 0, 0])
    cam_center = torch.tensor(cam_pose[:3, -1])
    v = get_unit_vector(start=cam_center, end=center)
    x, y, z = get_plane_points(v=v, p=v, h=H, w=W)  # image plane
    query_points, image_gt_colors = get_rays_camera_to_plane(
        image=image,
        cam_center=cam_center,
        plane=(x, y, z),
        tf=tf,
        tn=tn,
        N=N,
        M=M,
        H=H,
        W=W,
        mode=mode,
    )
    query_points = query_points.float()
    image_gt_colors = image_gt_colors.float()
    return query_points, image_gt_colors


def get_ray_vectors(
    cam_pose,
    focal_length,
    H=100,
    W=100,
):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32), torch.arange(H, dtype=torch.float32), indexing="xy")
    x = (i - H / 2) / focal_length
    y = -(j - W / 2) / focal_length  # down direction for y
    z = -torch.ones_like(x)

    dirs = torch.stack((x, y, z), -1)
    dirs = dirs @ cam_pose[:3, :3].T  # cam2word

    origin = cam_pose[:3, -1]
    return dirs, origin


def sample_ray_vecotrs(dirs, origin, N=100, M=20, tn=2, tf=6, H=100, W=100, mode="train"):
    t_step = (tf - tn) / M

    if mode == "train":
        samples_i = torch.randint(0, H, size=(N,))
        samples_j = torch.randint(0, W, size=(N,))
        dirs_sampled = dirs[samples_i, samples_j, :]  # (N, 3)
    else:
        ii = torch.arange(0, H, 1)
        jj = torch.arange(0, W, 1)
        samples_i, samples_j = torch.meshgrid(ii, jj, indexing="ij")
        dirs_sampled = dirs[samples_i, samples_j].reshape(-1, 3)  # (HxW, 3)

    all_ray_queries = []
    for d in dirs_sampled:
        queries_dir = []
        for t in torch.arange(tn, tf, t_step):
            query = origin + t * d
            queries_dir.append(torch.cat([query, d], -1))
        queries_dir = torch.stack(queries_dir, 0)
        all_ray_queries.append(queries_dir)
    all_ray_queries = torch.stack(all_ray_queries, 0)  # NxMx6
    return all_ray_queries, (samples_i, samples_j)


def plot_ray_vectors(
    dirs,
    origin,
    all_ray_queries,
    N=25,
    M=10,
    tn=2,
    tf=6,
):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    reshaped_dirs = dirs.reshape(-1, 3)

    ax.plot(origin[0], origin[1], origin[2], marker="o", color="r", alpha=1, markersize=10, label="Camera")
    ax.plot(0, 0, 0, marker="o", color="k", alpha=1, markersize=50, label="Object")

    for dir in reshaped_dirs:
        draw_parametric_vector(ax, origin, dir, tf, color="green")
    all_ray_queries = all_ray_queries.reshape(-1, 6)
    ax.scatter(all_ray_queries[:, 0], all_ray_queries[:, 1], all_ray_queries[:, 2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_aspect("equal")
    plt.show()


def plot_camera_ray_query(cam_pose):
    center = torch.tensor([0, 0, 0])
    batch_id = 0
    cam_center = torch.tensor(cam_pose[batch_id, :3, -1])
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # object
    x, y, z = get_sphere_points(center=center, radius=1)
    draw_surface(ax, x, y, z)

    plot_point(ax, p=cam_center)  # camera pos
    draw_vector_start_end(ax, start=cam_center, end=center)  # principal dir
    v = get_unit_vector(start=cam_center, end=center)
    x, y, z = get_plane_points(v=v, p=v, h=5, w=5)  # image plane
    draw_surface(ax, x, y, z)
    image = torch.zeros(5, 5, 3)
    all_ray_queries, _ = get_rays_camera_to_plane(image, cam_center, plane=(x, y, z), tn=1, tf=3, M=10, N=20, H=5, W=5)
    for p in all_ray_queries.reshape(-1, 6):
        plot_point(ax, p[:3])

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
    # plot_camera_ray_query(torch.tensor(data["poses"][2:3]))
    poses = torch.tensor(data["poses"][2])
    focal = torch.tensor(data["focal"])
    image = torch.tensor(data["images"][2])
    N = 10
    M = 10
    tn = 2
    tf = 6
    dirs, origin = get_ray_vectors(poses, focal, H=100, W=100)
    all_ray_queries, (samples_i, samples_j) = sample_ray_vecotrs(dirs, origin, N=N, M=M, tn=tn, tf=tf, mode="train")
    plot_ray_vectors(
        dirs[samples_i, samples_j],
        origin,
        all_ray_queries,
        N=N,
        M=M,
        tn=tn,
        tf=tf,
    )
    image_sampled = image[samples_i, samples_j]
