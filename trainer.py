import os
import torch
import logging

import pandas as pd

from BaseTrainerClass import TrainerClass

from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from mag.experiment import Experiment

try:
    from apex import amp
except ImportError:
    print("Please install apex from https://www.github.com/nvidia/apex")

writer = SummaryWriter()


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, batch_accumulation=1):
        self.batch_accumulation = batch_accumulation
        self.reset()

    def reset(self):
        self.current = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.current = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (self.count * self.batch_accumulation)


"""
=================================CREATING TRAINER CLASS=================================
"""


class Trainer:
    def __init__(self, model: TrainerClass, cfg):

        # we cannot have both schedulers active at the same time
        assert not (cfg["step_scheduler"] is True and cfg["validation_scheduler"] is True)

        self.optimizer, self.scheduler = model.get_optimizer_scheduler()

        self.model = model

        self.cfg = cfg
        self.description = cfg["description"]

        self.best_loss = 10000000
        self.best_validation_metric = 10000000
        self.current_epoch = 0
        self.metric_container = {}
        self.global_step = 0

        self.settings = {
            "batch_size": None,
            "epochs": 1,
            "steps_per_epoch": None,
            "validation_steps": None,
            "validation_batch_size": None,
            "validation_freq": 1,
            "checkpoint_every_n_steps": None,
            "shuffle": True,
        }

        self.__set_logger()

        if self.cfg["use_apex"]:
            opt_level = self.cfg["apex_opt_level"]
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)

        self.train_columns = ["epoch", "step", "current_loss"]
        self.validate_columns = ["epoch"]

        self.train_df = None
        self.valid_df = pd.DataFrame(columns=["epoch", "loss", "avg_metric"])

    def __set_logger(self):
        logging_dir = os.path.join(Path(__file__).parent.absolute(), self.cfg["experiment_path"], "info.log")

        logging.StreamHandler(stream=None)
        logger = logging.getLogger()

        fhandler = logging.FileHandler(filename=logging_dir, mode="a")
        formatter = logging.Formatter("%(asctime)s - %(process)d -  %(message)s")
        fhandler.setFormatter(formatter)
        logger.addHandler(fhandler)
        logger.setLevel(logging.INFO)

    def __get_learning_rate(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr += [param_group["lr"]]
        return lr

    def __initialize_csv_file(self, metrics_data):
        for i, (key, val) in enumerate(metrics_data):
            self.train_columns.append(key)
            self.metric_container[key] = AverageMeter()
            logging.info(f"adding metric {key}")
        self.train_columns.append("learning_rate")
        self.train_columns.append("saving_best")
        self.train_df = pd.DataFrame(columns=self.train_columns)

    def __write_to_tensorboard(self, metrics_data):
        # filling metrics/loss and adding to tensorboard
        for _, (key, val) in enumerate(metrics_data):
            self.metric_container[key].update(val, self.settings["batch_size"])
            if key == "loss":
                continue
            writer.add_scalar(f"Metric/{key}_{self.description}", self.metric_container[key].avg, self.global_step)
            writer.add_scalar(f"Metric/{key}_all_runs", self.metric_container[key].avg, self.global_step)

        # writer to tensorboard
        writer.add_scalar(f"Loss/train_{self.description}", self.metric_container["loss"].avg, self.global_step)

        writer.add_scalar("Loss/all_train_losses", self.metric_container["loss"].avg, self.global_step)

        # adding all learning rates to tensorboard
        for idx, lr_value in enumerate(self.__get_learning_rate()):
            writer.add_scalar(f"Learning_rates/lr_{idx}_{self.description}", lr_value, self.global_step)

    def __train_one_epoch(self, dataloader):
        epochs = self.settings["epochs"]
        self.model.model.train()
        torch.set_grad_enabled(True)

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
                self.__initialize_csv_file(metrics.items())

            if self.cfg["use_apex"]:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()

            if self.cfg["step_scheduler"]:
                self.scheduler.step()

            self.__write_to_tensorboard(metrics.items())

            # set pbar description
            current_loss = self.metric_container["loss"].current
            avg_loss = self.metric_container["loss"].avg
            pbar.set_description(
                f"TRAIN epoch {self.current_epoch+1}/{epochs} idx {step} \
                current loss {current_loss}, avg loss {avg_loss}"
            )

            # Saving interval - optional
            if (
                self.settings["checkpoint_every_n_steps"] is not None
                and (step + 1) % self.settings["checkpoint_every_n_steps"] == 0
            ):
                save_dir = os.path.join(self.cfg["save_path"], f"epoch_{self.current_epoch}")
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                save_file = os.path.join(save_dir, f"epoch_{self.current_epoch}_step_{step}_avg_loss_{avg_loss}.bin")
                self.save(save_file)

            # Saving best loss and registering best loss
            saving_best = False
            if self.best_loss > self.metric_container["loss"].avg:
                saving_best = True
                logging.info("saving best model with loss {}".format(self.metric_container["loss"].avg))
                self.best_loss = self.metric_container["loss"].avg
                torch.save(
                    self.model.model.state_dict(), os.path.join(self.cfg["save_path"], "best_model_checkpoint.pth")
                )
                self.cfg["experiment"].register_result("best_loss", self.metric_container["loss"].avg)

            # registering last loss
            self.cfg["experiment"].register_result("last_loss", self.metric_container["loss"].avg)

            # adding given step data to csv file
            new_row = (
                [self.current_epoch, step, loss.detach().item()]
                + [metric.avg for metric in self.metric_container.values()]
                + [self.__get_learning_rate(), saving_best]
            )

            new_series = pd.Series(new_row, index=self.train_df.columns)

            self.train_df = self.train_df.append(new_series, ignore_index=True)
            self.train_df.to_csv(os.path.join(self.cfg["experiment_path"], "train_logs.csv"), index=False)

        # Saving last checkpoint in epoch
        save_file = os.path.join(self.cfg["save_path"], "last_checkpoint.bin")
        self.save(save_file)

    def __validation_step(self, dataloader, validation_metric):
        epochs = self.settings["epochs"]
        validation_sum_loss = AverageMeter()
        val_metric = AverageMeter()
        self.model.model.eval()

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
                val_metric.update(metrics[validation_metric], self.settings["validation_batch_size"])

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
            self.valid_df.to_csv(os.path.join(self.cfg["experiment_path"], "validation_logs.csv"), index=False)
            logging.info(f"Validation result metric for epoch {self.current_epoch} = {val_metric.avg}")

            if self.best_validation_metric > val_metric.avg:
                file_name = f"best_model_validation_{self.description}.pth"

                logging.info(
                    "saving best model with epoch {} validation metric {} as {}".format(
                        self.current_epoch, val_metric.avg, file_name
                    )
                )
                self.best_validation_metric = val_metric.avg
                torch.save(self.model.model.state_dict(), os.path.join(self.cfg["save_path"], file_name))
                self.cfg["experiment"].register_result("best_validation_loss", validation_sum_loss.avg)
                self.cfg["experiment"].register_result("best_validation_metric", val_metric.avg)

    def __predict_outputs(self, dataloader):
        self.model.model.eval()

        output_values = []

        with torch.no_grad():
            pbar = tqdm(dataloader, total=len(dataloader), dynamic_ncols=True)
            for data in pbar:
                output = self.model.validation_step(data)

                output_values.append((output, data))

        return output_values

    def save(self, path):
        logging.info(f"saving checkpoint to {path}")
        self.model.model.eval()

        save_dict = {
            "model_state_dict": self.model.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "best_loss": self.best_loss,
        }

        if self.cfg["step_scheduler"] is not None:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.cfg["use_apex"]:
            save_dict["amp"] = amp.state_dict()

        torch.save(save_dict, path)

    def load(self, path):
        logging.info(f"loading checkpoint from {path}")
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.current_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint["best_loss"]

        if self.cfg["step_scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.cfg["use_apex"]:
            amp.load_state_dict(checkpoint["amp"])

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
            :param int epochs: NUmber of epochs to train
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

        assert self.model is not None
        assert self.optimizer is not None
        assert batch_size is not None
        assert train_dataset is not None

        self.settings["batch_size"] = batch_size
        self.settings["epochs"] = epochs
        self.settings["shuffle"] = shuffle

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

        for epoch in range(self.settings["epochs"]):
            logging.info("starting epoch {}/{} training step".format(epoch + 1, self.settings["epochs"]))
            self.current_epoch = epoch
            self.__train_one_epoch(train_dataloader)

            if validation_dataloader is not None and (epoch + 1) % self.settings["validation_freq"] == 0:
                self.__validation_step(validation_dataloader, validation_metric=validation_metric)

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
