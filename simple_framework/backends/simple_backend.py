from simple_framework.backends.BaseBackendClass import BackendBase

from simple_framework.trainer.BaseTrainerClass import SimpleFrameworkWrapper
from typing import Dict, List

from tqdm import tqdm
import logging
import pandas as pd
import os

import torch

from simple_framework.utilities.metrics import AverageMeter

from simple_framework.callbacks.CheckpointCallback import CheckpointCallback
from simple_framework.callbacks.CallbacksHandler import CallbacksHandler


class SimpleBackend(BackendBase):
    def __init__(self, model: SimpleFrameworkWrapper):
        super().__init__(model)

    def setup(self, settings: Dict, callbacks: List):
        self.optimizer, self.scheduler = self.model.get_optimizer_scheduler()
        self.current_epoch = 0
        self.metric_container = {}
        self.validation_metrics_container = {}
        self.current_step = 0
        self.global_step = 0
        self.settings = settings

        self.callback_handler = CallbacksHandler(callbacks)

        self.set_logger()

        self.callback_handler.handle(self, self.model, "on_setup")

    def train_step(self, data, step):

        self.callback_handler.handle(self, self.model, "on_train_step_start")
        self.optimizer.zero_grad()
        loss, metrics = self.model.train_step(data)

        # creating metrics container first time we see it
        if not self.metric_container:
            super().initialize_metrics_container(metrics.items())

        loss.backward()

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        for _, (key, val) in enumerate(metrics.items()):
            self.metric_container[key].update(val, self.settings["batch_size"])

        self.callback_handler.handle(self, self.model, "on_train_step_end")

        current_loss = self.metric_container["loss"].current

        """super().write_to_tensorboard(metrics.items())"""

        return current_loss

    def train_epoch(self, train_dataloader):
        self.model.train()
        torch.set_grad_enabled(True)
        self.current_step = 0

        self.callback_handler.handle(self, self.model, "on_train_epoch_start")

        epochs = self.settings["epochs"]

        dl_iter = iter(train_dataloader)

        pbar = tqdm(range(self.settings["steps_per_epoch"]), dynamic_ncols=True)

        for step in pbar:
            self.current_step = step
            self.global_step = step + self.current_epoch * self.settings["steps_per_epoch"]
            try:
                data = next(dl_iter)
            except StopIteration:
                dl_iter = iter(train_dataloader)
                data = next(dl_iter)

            current_loss = self.train_step(data, step)

            avg_loss = self.metric_container["loss"].avg

            # set pbar description
            pbar.set_description(
                f"TRAIN epoch {self.current_epoch+1}/{epochs} idx {step} \
                current loss {current_loss}, avg loss {avg_loss}"
            )
        self.callback_handler.handle(self, self.model, "on_train_epoch_end")

    def train_phase(self, dataset: torch.utils.data.Dataset, validation_dataset: torch.utils.data.Dataset):

        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=self.settings["shuffle"],
            batch_size=self.settings["batch_size"],
            num_workers=self.settings["num_workers"],
        )

        if self.settings["steps_per_epoch"] is None:
            self.settings["steps_per_epoch"] = len(train_dataloader)

        for epoch in range(self.settings["epochs"]):
            logging.info("starting epoch {}/{} training step".format(epoch + 1, self.settings["epochs"]))
            self.train_epoch(train_dataloader)

            if validation_dataset is not None and (epoch + 1) % self.settings["validation_freq"] == 0:
                self.validation_phase(validation_dataset)

            self.current_epoch += 1

    def validation_phase(self, dataset: torch.utils.data.Dataset):

        self.callback_handler.handle(self, self.model, "on_validation_epoch_start")

        validation_dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.settings["validation_batch_size"],
            num_workers=self.settings["num_workers"],
        )

        if self.settings["validation_steps"] is None:
            self.settings["validation_steps"] = len(validation_dataloader)

        epochs = self.settings["epochs"]
        validation_sum_loss = AverageMeter()
        val_metric = AverageMeter()
        self.validation_metric_container = {}
        self.model.eval()

        with torch.no_grad():
            dl_iter = iter(validation_dataloader)

            pbar = tqdm(range(self.settings["validation_steps"]), dynamic_ncols=True)

            for step in pbar:
                self.current_step = step
                self.callback_handler.handle(self, self.model, "on_validation_step_start")
                try:
                    data = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(validation_dataloader)
                    data = next(dl_iter)

                loss, metrics = self.model.train_step(data)

                # add to average meter for loss
                validation_sum_loss.update(loss.item(), self.settings["validation_batch_size"])

                # validation metric update
                val_metric.update(metrics[self.settings["validation_metric"]], self.settings["validation_batch_size"])

                for _, (key, val) in enumerate(metrics.items()):
                    self.validation_metrics_container[key].update(val, self.settings["batch_size"])

                # set pbar description
                pbar.set_description(
                    (
                        f"VALIDATION epoch {self.current_epoch+1}/{epochs} step {step}"
                        f" current loss  {validation_sum_loss.current}, avg loss {validation_sum_loss.avg}"
                        f", validation_metric {val_metric.avg}"
                    )
                )
                self.callback_handler.handle(self, self.model, "on_validation_step_end")

            logging.info(f"Validation result metric for epoch {self.current_epoch} = {val_metric.avg}")

            self.callback_handler.handle(self, self.model, "on_validation_epoch_end")
