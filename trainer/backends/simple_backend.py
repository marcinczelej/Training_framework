from backends.BaseBackendClass import BackendBase

from BaseTrainerClass import TrainerClass
from typing import Dict

from tqdm import tqdm
import logging
import pandas as pd
import os

import torch

from utilities.metrics import AverageMeter


class SimpleBackend(BackendBase):
    def __init__(self, model: TrainerClass):
        super().__init__(model)

    def train_phase(self, dataloader: torch.utils.data.DataLoader):
        epochs = self.settings["epochs"]
        self.model.train()
        torch.set_grad_enabled(True)

        self.checkpointer.set_epoch(self.current_epoch)

        dl_iter = iter(dataloader)

        pbar = tqdm(range(self.settings["steps_per_epoch"]), dynamic_ncols=True)

        for step in pbar:
            self.global_step = step + self.current_epoch * self.settings["steps_per_epoch"]
            try:
                data = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dataloader)
                data = next(dl_iter)

            self.optimizer.zero_grad()

            loss, metrics = self.model.train_step(data)

            # creating metrics container first time we see it
            if not self.metric_container:
                super().initialize_csv_file(metrics.items())

            loss.backward()

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            super().write_to_tensorboard(metrics.items())

            current_loss = self.metric_container["loss"].current
            avg_loss = self.metric_container["loss"].avg

            # Saving interval - optional
            if (
                self.settings["checkpoint_every_n_steps"] is not None
                and (step + 1) % self.settings["checkpoint_every_n_steps"] == 0
            ):
                self.checkpointer.save_checkpoint(
                    metric_value=avg_loss,
                    model=self.model,
                    checkpoint_name=f"epoch_{self.current_epoch}_step_{step}_avg_loss_{avg_loss}.bin",
                    checkpoint_path=os.path.join(self.settings["save_path"], f"epoch_{self.current_epoch}"),
                )

            # save best
            self.checkpointer.should_save_best(
                metric_value=self.metric_container["loss"].avg, model=self.model, task="train"
            )

            # registering last loss
            self.settings["experiment"].register_result("last_loss", self.metric_container["loss"].avg)

            # adding given step data to csv file
            new_row = (
                [self.current_epoch, step, loss.detach().item()]
                + [metric.avg for metric in self.metric_container.values()]
                + [super().get_learning_rate()]
            )

            new_series = pd.Series(new_row, index=self.train_df.columns)

            self.train_df = self.train_df.append(new_series, ignore_index=True)
            self.train_df.to_csv(os.path.join(self.settings["experiment_path"], "train_logs.csv"), index=False)

            # set pbar description
            pbar.set_description(
                f"TRAIN epoch {self.current_epoch+1}/{epochs} idx {step} \
                current loss {current_loss}, avg loss {avg_loss}"
            )

        # Saving last checkpoint in epoch
        self.checkpointer.save(
            checkpoint_name="last_checkpoint.bin",
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.current_epoch += 1

    def validation_phase(self, dataloader: torch.utils.data.DataLoader):
        epochs = self.settings["epochs"]
        validation_sum_loss = AverageMeter()
        val_metric = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            print("len dataloader ", len(dataloader))
            dl_iter = iter(dataloader)

            pbar = tqdm(range(self.settings["validation_steps"]), dynamic_ncols=True)

            for step in pbar:
                try:
                    data = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(dataloader)
                    data = next(dl_iter)

                loss, metrics = self.model.train_step(data)

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

            # adding given epoch data to csv file
            new_row = [self.current_epoch, validation_sum_loss.avg, val_metric.avg]
            new_series = pd.Series(new_row, index=self.valid_df.columns)

            self.valid_df = self.valid_df.append(new_series, ignore_index=True)
            self.valid_df.to_csv(os.path.join(self.settings["experiment_path"], "validation_logs.csv"), index=False)
            logging.info(f"Validation result metric for epoch {self.current_epoch} = {val_metric.avg}")

            # save best
            self.checkpointer.should_save_best(
                metric_value=val_metric.avg,
                checkpoint_name="best_model_validation_{}.pth".format(self.settings["description"]),
                model=self.model,
                task="validation",
            )
