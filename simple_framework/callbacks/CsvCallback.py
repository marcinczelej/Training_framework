from simple_framework.callbacks.BaseCallbackClass import CallbackBase
import torch
import os
import logging

import pandas as pd

from pathlib import Path


class CsvCallback(CallbackBase):
    def __init__(
        self,
        save_dir: str = "",
        train_csv_name: str = "train.csv",
        validation_csv_name: str = "validation.csv",
        save_training: bool = False,
        save_validation: bool = False,
        save_interval: str = "step",
    ):

        assert save_interval == "step" or save_interval == "epoch"

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        self.save_dir = save_dir
        self.train_csv_name = train_csv_name
        self.validation_csv_name = validation_csv_name
        self.save_training = save_training
        self.save_validation = save_validation
        self.save_interval = save_interval

        self.train_columns = ["epoch", "step", "current_loss"]
        self.validation_columns = ["epoch", "step"]
        self.initialized = False

        self.train_df = None
        self.valid_df = None

    def on_train_epoch_end(self, trainer, module):
        if self.save_training is not True or self.save_interval == "step":
            return

        if self.initialized is False:
            self.__initialize_csv_file(trainer)
            self.initialized = True

        self.__save_train_data(trainer)

    def on_train_step_end(self, trainer, module):
        if self.save_training is not True or self.save_interval == "epoch":
            return

        if self.initialized is False:
            self.__initialize_csv_file(trainer)
            self.initialized = True

        self.__save_train_data(trainer)

    def on_validation_epoch_end(self, trainer, module):
        if self.save_validation is not True or self.save_interval == "step":
            return

        if self.initialized is False:
            self.__initialize_csv_file(trainer)
            self.initialized = True

        self.__save_validation_data(trainer)

    def on_validation_step_end(self, trainer, module):
        if self.save_validation is not True or self.save_interval == "epoch":
            return

        if self.initialized is False:
            self.__initialize_csv_file(trainer)
            self.initialized = True

        print("saving")
        self.__save_validation_data(trainer)

    def __initialize_csv_file(self, trainer):
        for _, (key, _) in enumerate(trainer.metric_container.items()):
            self.train_columns.append(key)
            self.validation_columns.append(key)
        self.train_columns.append("learning_rate")
        self.train_df = pd.DataFrame(columns=self.train_columns)
        self.valid_df = pd.DataFrame(columns=self.validation_columns)

    def __save_train_data(self, trainer):
        # adding given step data to csv file
        new_row = (
            [trainer.current_epoch, trainer.current_step, trainer.metric_container["loss"].current]
            + [metric.avg for metric in trainer.metric_container.values()]
            + [trainer.get_learning_rate()]
        )
        new_series = pd.Series(new_row, index=self.train_df.columns)
        self.train_df = self.train_df.append(new_series, ignore_index=True)
        self.train_df.to_csv(os.path.join(self.save_dir, self.train_csv_name), index=False)

    def __save_validation_data(self, trainer):
        # adding given epoch data to csv file
        new_row = [trainer.current_epoch, trainer.current_step] + [
            metric.avg for metric in trainer.validation_metrics_container.values()
        ]
        new_series = pd.Series(new_row, index=self.valid_df.columns)

        self.valid_df = self.valid_df.append(new_series, ignore_index=True)
        self.valid_df.to_csv(os.path.join(self.save_dir, self.validation_csv_name), index=False)
