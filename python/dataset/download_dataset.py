import os
import zipfile
from pathlib import Path

import requests


def download_tiny_nerf_dataset(data_dir="data", force_download=False):
    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
    file_path = os.path.join(data_dir, "tiny_nerf_data.npz")

    if not os.path.exists(file_path) or force_download:
        response = requests.get(url)
        with open(file_path, "wb") as file:
            file.write(response.content)
        print("Downloaded tiny_nerf_data.npz to", data_dir)

    return Path(file_path)


def download_nerf_dataset(data_dir="data"):
    data_dir = Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip"
    response = requests.get(url)
    with open(os.path.join(data_dir, "nerf_example_data.zip"), "wb") as file:
        file.write(response.content)

    with zipfile.ZipFile(os.path.join(data_dir, "nerf_example_data.zip"), "r") as zip_file:
        zip_file.extractall(data_dir)
    return data_dir / "nerf_example_data.npz"
