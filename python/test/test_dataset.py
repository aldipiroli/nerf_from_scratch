import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.download_dataset import download_tiny_nerf_dataset


def test_download_tiny_nerf_dataset():
    path = download_tiny_nerf_dataset(force_download=True)
    assert path.is_file()
