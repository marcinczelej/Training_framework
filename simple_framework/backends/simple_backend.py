from simple_framework.backends.BaseBackendClass import BackendBase

from simple_framework.trainer.BaseTrainerClass import SimpleFrameworkWrapper
from typing import Dict, List

from tqdm import tqdm
import logging
import pandas as pd
import os

import torch

from simple_framework.utilities.metrics import AverageMeter
from simple_framework.utilities.checkpoint_saver import Checkpoint_saver

from simple_framework.callbacks.CheckpointCallback import CheckpointCallback
from simple_framework.callbacks.CallbacksHandler import CallbacksHandler


class SimpleBackend(BackendBase):
    def __init__(self, model: SimpleFrameworkWrapper):
        super().__init__(model)

    def setup(self, settings: Dict, callbacks: List):
        self.optimizer, self.scheduler = self.model.get_optimizer_scheduler()
        self.current_epoch = 0
        self.metric_container = {}
        self.current_step = 0
        self.global_step = 0
        self.settings = settings

        self.callback_handler = CallbacksHandler(callbacks)

        self.set_logger()

        self.train_columns = ["epoch", "step", "current_loss"]
        self.validate_columns = ["epoch"]

        self.train_df = None
        self.valid_df = pd.DataFrame(columns=["epoch", "loss", "avg_metric"])

    def train_step(self, data, step):

        self.callback_handler.handle(self, self.model, "on_train_step_start")
        self.optimizer.zero_grad()
        loss, metrics = self.model.train_step(data)

        # creating metrics container first time we see it
        if not self.metric_container:
            super().initialize_csv_file(metrics.items())

        loss.backward()

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        for _, (key, val) in enumerate(metrics.items()):
            self.metric_container[key].update(val, self.settings["batch_size"])

        self.callback_handler.handle(self, self.model, "on_train_step_end")

        current_loss = self.metric_container["loss"].current

        """super().write_to_tensorboard(metrics.items())

        current_loss = self.metric_container["loss"].current
        avg_loss = self.metric_container["loss"].avg

        # save best
        self.checkpointer.should_save_best(
            metric_value=self.metric_container["loss"].avg, model=self.model, task="train"
        )

        # adding given step data to csv file
        new_row = (
            [self.current_epoch, step, loss.detach().item()]
            + [metric.avg for metric in self.metric_container.values()]
            + [super().get_learning_rate()]
        )

        new_series = pd.Series(new_row, index=self.train_df.columns)

        self.train_df = self.train_df.append(new_series, ignore_index=True)
        self.train_df.to_csv(os.path.join(self.settings["experiment_path"], "train_logs.csv"), index=False)"""

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
            # self.checkpointer.set_epoch(self.current_epoch)
            self.train_epoch(train_dataloader)

            if validation_dataset is not None and (epoch + 1) % self.settings["validation_freq"] == 0:
                self.validation_phase(validation_dataset)

            self.current_epoch += 1

    def validation_phase(self, dataset: torch.utils.data.Dataset):

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
        self.model.eval()

        with torch.no_grad():
            self.callback_handler.handle(self, self.model, "on_validation_epoch_start")
            dl_iter = iter(validation_dataloader)

            pbar = tqdm(range(self.settings["validation_steps"]), dynamic_ncols=True)

            for step in pbar:
                self.current_step = step
                try:
                    data = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(validation_dataloader)
                    data = next(dl_iter)

                loss, metrics = self.model.train_step(data)
                self.callback_handler.handle(self, self.model, "on_validation_step_start")

                # add to average meter for loss
                validation_sum_loss.update(loss.item(), self.settings["validation_batch_size"])

                # validation metric update
                val_metric.update(metrics[self.settings["validation_metric"]], self.settings["validation_batch_size"])

                # set pbar description
                pbar.set_description(
                    (
                        f"VALIDATION epoch {self.current_epoch+1}/{epochs} step {step}"
                        f" current loss  {validation_sum_loss.current}, avg loss {validation_sum_loss.avg}"
                        f", validation_metric {val_metric.avg}"
                    )
                )
                self.callback_handler.handle(self, self.model, "on_validation_step_end")

            # adding given epoch data to csv file
            new_row = [self.current_epoch, validation_sum_loss.avg, val_metric.avg]
            new_series = pd.Series(new_row, index=self.valid_df.columns)

            self.valid_df = self.valid_df.append(new_series, ignore_index=True)
            self.valid_df.to_csv(os.path.join(self.settings["experiment_path"], "validation_logs.csv"), index=False)
            logging.info(f"Validation result metric for epoch {self.current_epoch} = {val_metric.avg}")

            """ # save best
            self.checkpointer.should_save_best(
                metric_value=val_metric.avg,
                checkpoint_name="best_model_validation_{}.pth".format(self.settings["description"]),
                model=self.model,
                task="validation",
            )"""
            self.callback_handler.handle(self, self.model, "on_validation_epoch_end")
