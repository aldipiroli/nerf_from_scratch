from dataset.tiny_nerf_dataset import TinyNeRFDataset
from model.nerf import NeRFModel
from model.nerf_loss import NeRFLoss
from utils.misc import get_logger, load_config
from utils.trainer import Trainer


def train():
    config = load_config("config/nerf_config.yaml")
    print(config)
    logger = get_logger(config["LOG_DIR"])
    trainer = Trainer(config, logger)

    model_cfg = config["MODEL"]
    model = NeRFModel(
        input_size=model_cfg["input_size"], output_size=model_cfg["output_size"], embed_size=model_cfg["embed_size"]
    )
    trainer.set_model(model)

    data_config = config["DATA"]
    train_dataset = TinyNeRFDataset(
        root_dir=data_config["root"],
        mode="train",
        N=data_config["N"],
        M=data_config["M"],
        tn=data_config["tn"],
        tf=data_config["tf"],
        H=data_config["H"],
        W=data_config["W"],
    )

    val_dataset = TinyNeRFDataset(
        root_dir=data_config["root"],
        mode="val",
        N=data_config["N"],
        M=data_config["M"],
        tn=data_config["tn"],
        tf=data_config["tf"],
        H=data_config["H"],
        W=data_config["W"],
    )

    trainer.set_dataset(train_dataset, val_dataset, data_config=config["DATA"])
    trainer.set_optimizer(optim_config=config["OPTIM"])
    trainer.set_loss_function(loss_fn=NeRFLoss())
    trainer.save_checkpoint()
    # trainer.overfit_one_batch()
    trainer.train()


if __name__ == "__main__":
    train()
