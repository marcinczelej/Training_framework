import os
import torch
import logging
import sys

import pandas as pd

from BaseTrainerClass import TrainerClass

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from mag.experiment import Experiment

from utilities.checkpoint_saver import Checkpoint_saver

from backends.simple_backend import SimpleBackend

try:
    from apex import amp
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex")
    amp = None


"""
=================================CREATING TRAINER CLASS=================================
"""


class Trainer:
    def __init__(self, model: TrainerClass, cfg):

        self.model = model
        self.settings = {
            "batch_size": None,
            "epochs": 1,
            "steps_per_epoch": None,
            "validation_steps": None,
            "validation_batch_size": None,
            "validation_freq": 1,
            "checkpoint_every_n_steps": None,
            "shuffle": True,
            "description": cfg["description"],
            "experiment": cfg["experiment"],
            "save_path": cfg["save_path"],
            "experiment_path": cfg["experiment_path"],
            "backbone": "Simple",
            "validation_metric": "loss",
        }

        self.__select_backbone()

    def __select_backbone(self):
        if self.settings["backbone"] == "Simple":
            self.processing_backend = SimpleBackend(self.model)
        else:
            pass

    def fit(
        self,
        train_dataset=None,
        batch_size=None,
        epochs=1,
        validation_dataset=None,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        checkpoint_every_n_steps=None,
        validation_metric="loss",
        shuffle=True,
        num_workers=0,
    ):
        """
        Method used for training model with following parameters:

        Parameters:
            :param torch.utils.data.Dataset train_dataset: Dataset for model.
            :param int batch_size: batch size that should be used during training/validation
            :param int epochs: Number of epochs to train
            :param torch.utils.data.Dataset validation_dataset: Dataset for validation data
            :param int steps_per_epoch: training steps that should be performed each epoch.
                If not specified whole training set would be used
            :param int validation_steps: validation steps that should be  performed each validation phase.
                If not specified, whole validation set would be used
            :param int validation_batch_size: Batch size for validation step.
                If not set, training batch size would be used
            :param int validation_freq:After how many epochs validation should be performed
            :param int checkpoint_every_n_steps: should be set if we want to save model in given timesteps interval
            :param str validation_metric: metric that should be checkd after validation to see if result is better
            :param bool shuffle: should dataset be shuffled during training
            :param int num_workers: number of workers to use during training

        Returns:
            None
        """

        assert batch_size is not None
        assert train_dataset is not None

        self.settings["batch_size"] = batch_size
        self.settings["epochs"] = epochs
        self.settings["shuffle"] = shuffle
        self.settings["validation_metric"] = validation_metric

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers
        )

        if steps_per_epoch is None:
            self.settings["steps_per_epoch"] = len(train_dataloader)
        else:
            self.settings["steps_per_epoch"] = steps_per_epoch

        if validation_batch_size is None:
            self.settings["validation_batch_size"] = batch_size
        else:
            self.settings["validation_batch_size"] = validation_batch_size

        if validation_dataset is not None:
            validation_dataloader = torch.utils.data.DataLoader(
                validation_dataset,
                shuffle=shuffle,
                batch_size=self.settings["validation_batch_size"],
                num_workers=num_workers,
            )
            self.settings["validation_steps"] = len(validation_dataloader)
            self.settings["validation_freq"] = validation_freq

        self.settings["checkpoint_every_n_steps"] = checkpoint_every_n_steps

        logging.info("Training settings:")
        for _, (key, val) in enumerate(self.settings.items()):
            logging.info(f"{key} : {val}")

        """
        ========================== TRAINING VALIDATION PHASE ==========================
        """

        self.processing_backend.setup(self.settings)
        for epoch in range(self.settings["epochs"]):
            logging.info("starting epoch {}/{} training step".format(epoch + 1, self.settings["epochs"]))
            self.processing_backend.train_phase(train_dataloader)

            if validation_dataloader is not None and (epoch + 1) % self.settings["validation_freq"] == 0:
                self.processing_backend.validation_phase(validation_dataloader)

    def predict(self, dataloader):
        """
        Method used for predicting output given input data

        Parameters:
            :param DataLoader dataloader: Dataloader for given data

        Returns:
            prediction_output: Data in format: tuple(model_output, data for whitch given output was prodiced)
        """

        assert self.model is not None

        prediction_output = self.__predict_outputs(dataloader)

        return prediction_output
