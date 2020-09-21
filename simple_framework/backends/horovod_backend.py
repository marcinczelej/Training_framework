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

import horovod.torch as hvd


class HorovodBackend(BackendBase):
    def __init__(self, model: SimpleFrameworkWrapper):
        super().__init__(model)

    def setup(self, settings: Dict, callbacks: List):

        hvd.init()
        self.optimizer, self.scheduler = self.model.get_optimizer_scheduler()
        self.current_epoch = 0
        self.metric_container = {}
        self.validation_metrics_container = {}
        self.current_step = 0
        self.global_step = 0
        self.settings = settings

        self.callback_handler = CallbacksHandler(callbacks)

        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
            self.model.cuda(hvd.local_rank())

        print(self.get_learning_rate())

        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= hvd.size()

        print(self.get_learning_rate())

        if self.scheduler is not None:
            for idx, lr in enumerate(self.scheduler.base_lrs):
                logging.info(f"Base scheduler learning rate #{idx} : {lr}")

            self.scheduler.base_lrs = [lr * hvd.size() for lr in self.scheduler.base_lrs]

            for idx, lr in enumerate(self.scheduler.base_lrs):
                logging.info(f"Scheduler learning rate #{idx} : {lr} after resizing to match horovod GPU number")

        hvd.broadcast_parameters(self.model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(self.optimizer, root_rank=0)

        compression = hvd.Compression.fp16 if self.settings["use_fp16"] else hvd.Compression.none

        def get_named_parameters(model, optimizer):
            opt_params = {p for group in optimizer.param_groups for p in group.get("params", [])}
            return [(name, p) for name, p in model.named_parameters() if p in opt_params]

        hvd.DistributedOptimizer(
            self.optimizer, named_parameters=get_named_parameters(self.model, self.optimizer), compression=compression
        )

        self.callback_handler.handle(self, self.model, "on_setup")

    def train_step(self, data, step):
        # Horovod: print logs on the first worker.
        verbose = 1 if hvd.rank() == 0 else 0

        if verbose:
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

        if verbose:
            for _, (key, val) in enumerate(metrics.items()):
                self.metric_container[key].update(val, self.settings["batch_size"])

            self.callback_handler.handle(self, self.model, "on_train_step_end")

        current_loss = self.metric_container["loss"].current

        return current_loss

    def train_epoch(self, train_dataloader):
        self.model.train()
        torch.set_grad_enabled(True)
        self.current_step = 0

        self.callback_handler.handle(self, self.model, "on_train_epoch_start")

        epochs = self.settings["epochs"]

        dl_iter = iter(train_dataloader)

        pbar = tqdm(range(self.settings["steps_per_epoch"]), dynamic_ncols=True, disable=(hvd.rank() != 0))

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
                f"TRAIN hvd rank: {hvd.rank()} epoch {self.current_epoch+1}/{epochs} idx {step} \
                current loss {current_loss}, avg loss {avg_loss}"
            )

        hvd.join()
        if hvd.rank() == 0:
            self.callback_handler.handle(self, self.model, "on_train_epoch_end")

    def train_phase(self, train_dataset: torch.utils.data.Dataset, validation_dataset: torch.utils.data.Dataset):

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=False,
            batch_size=self.settings["batch_size"],
            num_workers=self.settings["num_workers"],
            sampler=train_sampler,
        )

        if self.settings["steps_per_epoch"] is None:
            self.settings["steps_per_epoch"] = len(train_dataloader)

        for epoch in range(self.settings["epochs"]):
            if hvd.rank() == 0:
                logging.info("starting epoch {}/{} training step".format(epoch + 1, self.settings["epochs"]))
            train_sampler.set_epoch(self.current_epoch)
            self.train_epoch(train_dataloader)

            if validation_dataset is not None and (epoch + 1) % self.settings["validation_freq"] == 0:
                self.validation_phase(validation_dataset)

            if hvd.rank() == 0:
                self.current_epoch += 1

    def validation_phase(self, dataset: torch.utils.data.Dataset):

        self.callback_handler.handle(self, self.model, "on_validation_epoch_start")

        validation_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )

        validation_dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.settings["validation_batch_size"],
            num_workers=self.settings["num_workers"],
            sampler=validation_sampler,
        )

        if self.settings["validation_steps"] is None:
            self.settings["validation_steps"] = len(validation_dataloader)

        epochs = self.settings["epochs"]
        validation_sum_loss = AverageMeter()
        val_metric = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            dl_iter = iter(validation_dataloader)

            pbar = tqdm(range(self.settings["validation_steps"]), dynamic_ncols=True, disable=(hvd.rank() != 0))

            for step in pbar:
                self.current_step = step
                if hvd.rank() == 0:
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
                if hvd.rank() == 0:
                    val_metric.update(
                        metrics[self.settings["validation_metric"]], self.settings["validation_batch_size"]
                    )

                    for _, (key, val) in enumerate(metrics.items()):
                        self.validation_metrics_container[key].update(val, self.settings["batch_size"])

                # set pbar description
                pbar.set_description(
                    (
                        f"VALIDATION hvd rank: {hvd.rank()} epoch {self.current_epoch+1}/{epochs} step {step}"
                        f" current loss  {validation_sum_loss.current}, avg loss {validation_sum_loss.avg}"
                        f", validation_metric {val_metric.avg}"
                    )
                )
                hvd.join()
                if hvd.rank() == 0:
                    self.callback_handler.handle(self, self.model, "on_validation_step_end")

            hvd.join()
            if hvd.rank() == 0:
                logging.info(f"Validation result metric for epoch {self.current_epoch} = {val_metric.avg}")
                self.callback_handler.handle(self, self.model, "on_validation_epoch_end")
