import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from utils.render import get_orthogonal_vector, get_unit_vector, resize_image


def test_get_unit_vector():
    v1 = torch.tensor([1, 0, 0]).float()
    v2 = torch.tensor([10, 0, 0]).float()
    unit_v = get_unit_vector(start=v1, end=v2)
    target = torch.tensor([2, 0, 0]).float()
    assert torch.allclose(unit_v, target)


def test_get_orthogonal_vectors():
    v1 = torch.tensor([1, 0, 0]).float()
    v2 = get_orthogonal_vector(v1)
    dot_prod = torch.dot(v1, v2)
    target = torch.zeros(3).float()
    assert torch.allclose(dot_prod, target)


def test_resize_image():
    img = torch.zeros(100, 100, 3).float()
    h, w = 50, 50
    img_resize = resize_image(img, h, w)
    assert img_resize.shape == (h, w, 3)


if __name__ == "__main__":
    test_get_unit_vector()
    test_get_orthogonal_vectors()
    print("All test passed!")
