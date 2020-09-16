import torch
from pathlib import Path

import os
import logging


class Checkpoint_saver:
    def __init__(self, checkpoints_dir, experiment, description):
        self.best_train_metric = 10000000
        self.best_validation_metric = 10000000
        self.current_epoch = 0
        self.descritpion = description
        self.checkpoints_dir = checkpoints_dir
        self.experiment = experiment

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def should_save_best(self, metric_value, model, checkpoint_name="best_model_checkpoint.pth", task="train"):

        metric_val = self.best_train_metric if task == "train" else self.best_validation_metric

        if metric_val > metric_value:
            if task == "train":
                self.best_train_metric = metric_value
            else:
                self.best_validation_metric = metric_value

            self.save_checkpoint(metric_value, model, checkpoint_name, task)

    def save_checkpoint(self, metric_value, model, checkpoint_name, task="train", checkpoint_path=None):

        if checkpoint_path is None:
            checkpoint_path = self.checkpoints_dir

        Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"saving model for {task} with metric {metric_value} as {checkpoint_name}")
        torch.save(model.state_dict(), os.path.join(checkpoint_path, checkpoint_name))
        self.experiment.register_result(f"best_{task}_metric", metric_value)

    def save(self, checkpoint_name, model, optimizer, scheduler=None, amp=None):

        path = os.path.join(self.checkpoints_dir, "last_checkpoint.bin")
        logging.info(f"saving checkpoint to {path}")
        model.model.eval()

        save_dict = {
            "model_state_dict": model.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_loss": self.best_validation_metric,
        }

        if scheduler is not None:
            save_dict["scheduler_state_dict"] = scheduler.state_dict()

        if amp is not None:
            save_dict["amp"] = amp.state_dict()

        torch.save(save_dict, path)

    def load(self, path, model, optimizer, scheduler=None, amp=None):
        logging.info(f"loading checkpoint from {path}")
        checkpoint = torch.load(path)
        model.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.best_validation_metric = checkpoint["best_loss"]

        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if amp is not None:
            amp.load_state_dict(checkpoint["amp"])

        return model, optimizer, self.current_epoch, scheduler, amp
