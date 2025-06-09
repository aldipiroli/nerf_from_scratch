from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.misc import get_device, plot_images
from utils.render import get_volume_rendering


class Trainer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.epoch = 0
        self.pred_threshold = 0.5

        self.ckpt_dir = Path(config["CKPT_DIR"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device()
        self.artifacts_img_dir = Path(config["IMG_OUT_DIR"])
        self.artifacts_img_dir.mkdir(parents=True, exist_ok=True)
        self.step = (config["DATA"]["tf"] - config["DATA"]["tn"]) / config["DATA"]["M"]

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.logger.info("Model:")
        self.logger.info(self.model)

    def save_checkpoint(self):
        model_path = Path(self.ckpt_dir) / f"ckpt_{str(self.epoch).zfill(4)}.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            model_path,
        )
        self.logger.info(f"Saved checkpoint in: {model_path}")

    def load_latest_checkpoint(self):
        if not self.ckpt_dir.exists():
            self.logger.info("No checkpoint directory found.")
            return None

        ckpt_files = sorted(self.ckpt_dir.glob("ckpt_*.pt"))
        if not ckpt_files:
            self.logger.info("No checkpoints found.")
            return None

        latest_ckpt = max(ckpt_files, key=lambda x: int(x.stem.split("_")[1]))
        self.logger.info(f"Loading checkpoint: {latest_ckpt}")

        checkpoint = torch.load(latest_ckpt, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        return latest_ckpt

    def set_dataset(self, train_dataset, val_dataset, data_config):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.data_config = data_config

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=data_config["batch_size"],
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
        )
        self.logger.info(f"Train Dataset: {self.train_dataset}")
        self.logger.info(f"Val Dataset: {self.val_dataset}")

    def set_optimizer(self, optim_config):
        self.optim_config = optim_config
        if self.optim_config["optimizer"] == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optim_config["lr"])

            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.optim_config.get("step_size", self.optim_config["weight_decay_step"]),
                gamma=self.optim_config.get("gamma", self.optim_config["weight_decay"]),
            )
        else:
            raise ValueError("Unknown optimizer")

        self.logger.info(f"Optimizer: {self.optimizer}")

    def set_loss_function(self, loss_fn):
        self.loss_fn = loss_fn.to(self.device)
        self.logger.info(f"Loss function {self.loss_fn}")

    def train(self, eval_every=50):
        for curr_epoch in range(self.optim_config["num_epochs"]):
            self.epoch = curr_epoch

            self.train_one_epoch()
            if (curr_epoch + 1) % eval_every == 0:
                self.evaluate_model()
                self.save_checkpoint()

    def train_one_epoch(self):
        self.model.train()
        with tqdm(enumerate(self.train_loader), desc=f"Epoch {self.epoch}") as pbar:
            for n_iter, (query_points, image_gt_colors) in pbar:
                self.optimizer.zero_grad()
                query_points = query_points.to(self.device)
                image_gt_colors = image_gt_colors.to(self.device)

                preds_color, preds_density = self.model(query_points)
                predicted_ray_colors = get_volume_rendering(preds_color, preds_density, step=self.step)

                loss = self.loss_fn(image_gt_colors, predicted_ray_colors)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_postfix({"loss": loss.item()})
        # self.save_checkpoint()

    def overfit_one_batch(self):
        from dataset.tiny_nerf_dataset import TinyNeRFDataset

        tn = 2
        tf = 6
        M = 32
        train_dataset = TinyNeRFDataset(
            root_dir="../data",
            mode="val",
            N=100 * 100,
            M=32,
            tn=2,
            tf=6,
            H=100,
            W=100,
        )
        step = (tf - tn) / M
        data_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
        )
        self.model.train()
        data_iter = iter(data_loader)
        query_points, image_gt_colors = next(data_iter)

        for i in range(1000000):
            self.optimizer.zero_grad()
            query_points = query_points.to(self.device)
            image_gt_colors = image_gt_colors.to(self.device)

            preds_color, preds_density = self.model(query_points)
            predicted_ray_colors = get_volume_rendering(preds_color, preds_density, step=step)

            loss = self.loss_fn(image_gt_colors, predicted_ray_colors)
            loss.backward()
            self.optimizer.step()
            print(f"iter {i}, loss {loss}")

            if i % 25 == 0:
                pred = predicted_ray_colors[0].reshape(100, 100, 3)
                gt = image_gt_colors[0].reshape(100, 100, 3)
                plot_images([pred, gt], filename=f"{self.artifacts_img_dir}/img.png", curr_iter=i)

    def evaluate_model(self, max_num_samples=3):
        self.logger.info("Running Evaluation...")
        self.model.eval()
        for i, (query_points, image_gt_colors) in enumerate(self.val_loader):
            if i > max_num_samples:
                break
            query_points = query_points.to(self.device)
            image_gt_colors = image_gt_colors.to(self.device)
            preds_color, preds_density = self.model(query_points)
            predicted_ray_colors = get_volume_rendering(preds_color, preds_density, step=self.step)
            loss = self.loss_fn(image_gt_colors, predicted_ray_colors)

            pred = predicted_ray_colors[0].reshape(100, 100, 3)
            gt = image_gt_colors[0].reshape(100, 100, 3)
            plot_images([pred, gt], filename=f"{self.artifacts_img_dir}/img_{str(i).zfill(3)}.png", curr_iter=i)
            print(f"Validation loss: {loss}")

    def gradient_sanity_check(self):
        total_gradient = 0
        no_grad_name = []
        grad_name = []
        for name, param in self.model.named_parameters():
            if param.grad is None:
                no_grad_name.append(name)
                self.logger.info(f"None grad: {name}")
            else:
                grad_name.append(name)
                total_gradient += torch.sum(torch.abs(param.grad))
        assert total_gradient == total_gradient
        if len(no_grad_name) > 0:
            self.logger.info(f"no_grad_name {no_grad_name}")
            raise ValueError("layers without gradient are present")
        assert len(no_grad_name) == 0
