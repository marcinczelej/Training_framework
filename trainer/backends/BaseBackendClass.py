from abc import ABC, abstractmethod
import torch
import os
import logging
import pandas as pd

from pathlib import Path

from BaseTrainerClass import TrainerClass
from typing import Dict

from utilities.checkpoint_saver import Checkpoint_saver
from utilities.metrics import AverageMeter

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class BackendBase(ABC):
    def __init__(self, model: TrainerClass):
        self.model = model

    def setup(self, settings: Dict):
        self.optimizer, self.scheduler = self.model.get_optimizer_scheduler()
        self.current_epoch = 0
        self.metric_container = {}
        self.global_step = 0
        self.settings = settings

        self.checkpointer = Checkpoint_saver(
            checkpoints_dir=self.settings["save_path"],
            experiment=self.settings["experiment"],
            description=self.settings["description"],
        )

        self.set_logger()

        self.train_columns = ["epoch", "step", "current_loss"]
        self.validate_columns = ["epoch"]

        self.train_df = None
        self.valid_df = pd.DataFrame(columns=["epoch", "loss", "avg_metric"])

        print(self.settings)

    @abstractmethod
    def train_phase(self, dataloader):
        raise NotImplementedError

    @abstractmethod
    def validation_phase(self, dataloader):
        raise NotImplementedError

    def set_logger(self):
        logging_dir = os.path.join(self.settings["experiment_path"], "info.log")

        logging.StreamHandler(stream=None)
        logger = logging.getLogger()

        fhandler = logging.FileHandler(filename=logging_dir, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(process)d -  %(message)s")
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.INFO)

    def initialize_csv_file(self, metrics_data):
        for i, (key, val) in enumerate(metrics_data):
            self.train_columns.append(key)
            self.metric_container[key] = AverageMeter()
            logging.info(f"adding metric {key}")
        self.train_columns.append("learning_rate")
        self.train_df = pd.DataFrame(columns=self.train_columns)

    def get_learning_rate(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr += [param_group["lr"]]
        return lr

    def write_to_tensorboard(self, metrics_data):
        # filling metrics/loss and adding to tensorboard

        description = self.settings["description"]

        for _, (key, val) in enumerate(metrics_data):
            self.metric_container[key].update(val, self.settings["batch_size"])
            if key == "loss":
                continue
            writer.add_scalar(f"Metric/{key}_{description}", self.metric_container[key].avg, self.global_step)
            writer.add_scalar(f"Metric/{key}_all_runs", self.metric_container[key].avg, self.global_step)

        # writer to tensorboard
        writer.add_scalar(f"Loss/train_{description}", self.metric_container["loss"].avg, self.global_step)

        writer.add_scalar("Loss/all_train_losses", self.metric_container["loss"].avg, self.global_step)

        # adding all learning rates to tensorboard
        for idx, lr_value in enumerate(self.get_learning_rate()):
            writer.add_scalar(f"Learning_rates/lr_{idx}_{description}", lr_value, self.global_step)
