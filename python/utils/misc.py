import logging
import os
from datetime import datetime

import torch
import yaml


def get_device():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    return device


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"log_{now}.log")
    logger = logging.getLogger(f"logger_{now}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger
