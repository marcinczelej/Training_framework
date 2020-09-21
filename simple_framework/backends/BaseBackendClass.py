from abc import ABC, abstractmethod
import torch
import os
import logging
import pandas as pd

from pathlib import Path

from simple_framework.trainer.BaseTrainerClass import SimpleFrameworkWrapper
from typing import Dict

from simple_framework.utilities.metrics import AverageMeter

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


class BackendBase(ABC):
    def __init__(self, model: SimpleFrameworkWrapper):
        self.model = model

    @abstractmethod
    def setup(self, settings: Dict):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, data, step):
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self, data):
        raise NotImplementedError

    @abstractmethod
    def train_phase(self, dataset: torch.utils.data.Dataset):
        raise NotImplementedError

    @abstractmethod
    def validation_phase(self, dataset: torch.utils.data.Dataset):
        raise NotImplementedError

    def set_logger(self):
        logging_dir = os.path.join(self.settings["experiment_path"], "info.log")

        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        logger = logging.getLogger()

        fhandler = logging.FileHandler(filename=logging_dir, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(process)d -  %(message)s")
        fhandler.setFormatter(formatter)
        fhandler.setLevel(logging.INFO)

        logger.addHandler(fhandler)
        logger.addHandler(ch)

    def initialize_metrics_container(self, metrics_data):
        for i, (key, val) in enumerate(metrics_data):
            self.metric_container[key] = AverageMeter()
            self.validation_metrics_container[key] = AverageMeter()
            logging.info(f"adding metric {key}")

    def get_learning_rate(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr += [param_group["lr"]]
        return lr

    def write_to_tensorboard(self, metrics_data):
        # filling metrics/loss and adding to tensorboard

        description = self.settings["description"]

        for _, (key, val) in enumerate(metrics_data):
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
