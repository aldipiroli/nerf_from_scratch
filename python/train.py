from dataset.tiny_nerf_dataset import TinyNeRFDataset
from model.nerf import TinyNerf
from model.nerf_loss import NeRFLoss
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


def train():
    config = load_config("config/nerf_config.yaml")
    print(config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model_cfg = config["MODEL"]
    model = TinyNerf(input_size=model_cfg["input_size"], output_size=model_cfg["output_size"])
    trainer.set_model(model)

    data_config = config["DATA"]
    train_dataset = TinyNeRFDataset(
        root_dir=data_config["root"],
        split="train",
        N=data_config["N"],
        M=data_config["M"],
        tn=data_config["tn"],
        tf=data_config["tf"],
        img_plane_h=data_config["img_plane_h"],
        img_plane_w=data_config["img_plane_w"],
    )

    val_dataset = TinyNeRFDataset(
        root_dir=data_config["root"],
        split="val",
        N=data_config["N"],
        M=data_config["M"],
        tn=data_config["tn"],
        tf=data_config["tf"],
        img_plane_h=data_config["img_plane_h"],
        img_plane_w=data_config["img_plane_w"],
    )

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss_fn=NeRFLoss())
    trainer.save_checkpoint()
    trainer.train()


if __name__ == "__main__":
    train()
