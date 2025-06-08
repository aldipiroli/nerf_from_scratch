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

    train_dataset = TinyNeRFDataset(root_dir=config["DATA"]["root"], split="train")
    val_dataset = TinyNeRFDataset(root_dir=config["DATA"]["root"], split="val")
    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss_fn=NeRFLoss())
    trainer.save_checkpoint()
    trainer.train()


if __name__ == "__main__":
    train()
