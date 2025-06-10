import argparse
from pathlib import Path

import torch
from dataset.tiny_nerf_dataset import TinyNeRFDataset
from model.nerf import NeRFModel
from torch.utils.data import DataLoader
from utils.misc import get_device, get_logger, load_config
from utils.render import reconstruct_object


def evaluate(ckpt_path):
    config = load_config("config/nerf_config.yaml")
    logger = get_logger(config["LOG_DIR"])
    logger.info(f"Config: {config}")
    device = get_device()

    model_cfg = config["MODEL"]
    model = NeRFModel(
        input_size=model_cfg["input_size"], output_size=model_cfg["output_size"], embed_size=model_cfg["embed_size"]
    )
    model = model.to(device)

    checkpoint = torch.load(ckpt_path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info(f"Loaded Model: {ckpt_path}")

    reconstruct_dataset = TinyNeRFDataset(
        root_dir=config["DATA"]["root"],
        mode="render",
    )
    data_loader = DataLoader(
        reconstruct_dataset,
        batch_size=1,
        shuffle=False,
    )
    step = (config["DATA"]["tf"] - config["DATA"]["tn"]) / config["DATA"]["M"]
    reconstruct_object(data_loader, model, step=step, device=device, artifacts_folder=config["IMG_OUT_DIR"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    evaluate(Path(args.ckpt))
