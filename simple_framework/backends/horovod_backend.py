from simple_framework.backends.BaseBackendClass import BackendBase

from simple_framework.trainer.BaseTrainerClass import TrainerClass
from typing import Dict

from tqdm import tqdm
import logging
import pandas as pd
import os

import torch

from simple_framework.utilities.metrics import AverageMeter
from simple_framework.utilities.checkpoint_saver import Checkpoint_saver

import horovod.torch as hvd


class HorovodBackend(BackendBase):
    def __init__(self, model: TrainerClass):
        super().__init__(model)

    def setup(self, settings: Dict):

        hvd.init()
        self.optimizer, self.scheduler = self.model.get_optimizer_scheduler()
        self.current_epoch = 0
        self.metric_container = {}
        self.global_step = 0
        self.settings = settings

        if torch.cuda.is_available():
            torch.cuda.set_device(hvd.local_rank())
            self.model.cuda(hvd.local_rank())

        self.checkpointer = Checkpoint_saver(
            checkpoints_dir=self.settings["save_path"],
            description=self.settings["description"],
        )

        self.set_logger()

        self.train_columns = ["epoch", "step", "current_loss"]
        self.validate_columns = ["epoch"]

        self.train_df = None
        self.valid_df = pd.DataFrame(columns=["epoch", "loss", "avg_metric"])

        print(self.get_learning_rate())

        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= hvd.size()

        print(self.get_learning_rate())

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

    def train_step(self, data, step):
        # Horovod: print logs on the first worker.
        verbose = 1 if hvd.rank() == 0 else 0

        self.optimizer.zero_grad()
        loss, metrics = self.model.train_step(data)

        # creating metrics container first time we see it
        if not self.metric_container:
            super().initialize_csv_file(metrics.items())

        loss.backward()

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        if verbose:
            for _, (key, val) in enumerate(metrics.items()):
                self.metric_container[key].update(val, self.settings["batch_size"])

            super().write_to_tensorboard(metrics.items())

        current_loss = self.metric_container["loss"].current
        avg_loss = self.metric_container["loss"].avg

        # Saving interval - optional
        if (
            self.settings["checkpoint_every_n_steps"] is not None
            and (step + 1) % self.settings["checkpoint_every_n_steps"] == 0
            and verbose
        ):
            self.checkpointer.save_checkpoint(
                metric_value=avg_loss,
                model=self.model,
                checkpoint_name=f"epoch_{self.current_epoch}_step_{step}_avg_loss_{avg_loss}.bin",
                checkpoint_path=os.path.join(self.settings["save_path"], f"epoch_{self.current_epoch}"),
            )

        # save best
        if verbose:
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
            self.train_df.to_csv(os.path.join(self.settings["experiment_path"], "train_logs.csv"), index=False)

        return current_loss

    def train_epoch(self, train_dataloader):

        self.model.train()
        torch.set_grad_enabled(True)

        epochs = self.settings["epochs"]

        dl_iter = iter(train_dataloader)

        pbar = tqdm(range(self.settings["steps_per_epoch"]), dynamic_ncols=True, disable=(hvd.rank() != 0))

        for step in pbar:
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

        # Saving last checkpoint in epoch
        if hvd.rank() == 0:
            self.checkpointer.save(
                checkpoint_name="last_checkpoint.bin",
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
            )
        hvd.join()

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
            self.checkpointer.set_epoch(self.current_epoch)
            self.train_epoch(train_dataloader)

            if validation_dataset is not None and (epoch + 1) % self.settings["validation_freq"] == 0:
                self.validation_phase(validation_dataset)

            if hvd.rank() == 0:
                self.current_epoch += 1

    def validation_phase(self, dataset: torch.utils.data.Dataset):

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

                # set pbar description
                pbar.set_description(
                    (
                        f"VALIDATION hvd rank: {hvd.rank()} epoch {self.current_epoch+1}/{epochs} step {step}"
                        f" current loss  {validation_sum_loss.current}, avg loss {validation_sum_loss.avg}"
                        f", validation_metric {val_metric.avg}"
                    )
                )

            if hvd.rank() == 0:
                # adding given epoch data to csv file
                new_row = [self.current_epoch, validation_sum_loss.avg, val_metric.avg]
                new_series = pd.Series(new_row, index=self.valid_df.columns)

                self.valid_df = self.valid_df.append(new_series, ignore_index=True)
                self.valid_df.to_csv(
                    os.path.join(self.settings["experiment_path"], "validation_logs.csv"), index=False
                )
                logging.info(f"Validation result metric for epoch {self.current_epoch} = {val_metric.avg}")

                # save best
                self.checkpointer.should_save_best(
                    metric_value=val_metric.avg,
                    checkpoint_name="best_model_validation_{}.pth".format(self.settings["description"]),
                    model=self.model,
                    task="validation",
                )
            hvd.join()
