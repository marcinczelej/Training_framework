from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv

import os
import sys
import shutil
import argparse
import math

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from pathlib import Path

from typing import Dict

from tqdm import tqdm
from torch import Tensor

from ranger import Ranger

import bz2
import pickle

import torchvision.models as models

from mag.experiment import Experiment
from schedulers import get_flat_cosine_schedule

from trainer.MainTrainerClass import Trainer


from sklearn.model_selection import train_test_split
from collections import OrderedDict

from BaseTrainerClass import TrainerClass


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print(torch.cuda.is_available())

"""
=================================PARAMS=================================

"""

Params = {
    "step_scheduler": True,
    "epoch_scheduler": False,
    "EPOCHS": 1,
    "optimizer": "Ranger_flat_cosine",
    "learning_rate": 1e-4,
    "net_type": "Resnet18",
}

trainer_params = {
    "data_path": "data",
    "save_path": "checkpoints",
    "checkpoint_path": "checkpoints/best_model_checkpoint.pth",
    "experiment_path": "data",
    "experiment": None,
    "apex_opt_level": "01",
    "use_apex": False,
    "description": "",
    "step_scheduler": True,
    "validation_scheduler": False,
}

"""
=================================TRAIN CONFIG=================================

"""

train_cfg = {
    "format_version": 4,
    "model_params": {
        "model_architecture": "resnet50",
        "history_num_frames": 10,
        "history_step_size": 1,
        "history_delta_time": 0.1,
        "future_num_frames": 50,
        "future_step_size": 1,
        "future_delta_time": 0.1,
    },
    "raster_params": {
        "raster_size": [400, 400],
        "pixel_size": [0.5, 0.5],
        "ego_center": [0.25, 0.5],
        "map_type": "py_semantic",
        "satellite_map_key": "aerial_map/aerial_map.png",
        "semantic_map_key": "semantic_map/semantic_map.pb",
        "dataset_meta_key": "meta.json",
        "filter_agents_threshold": 0.5,
    },
    "data_loader_data": {"key": "scenes/train.zarr", "batch_size": 4, "shuffle": True, "num_workers": 0},
    "train_params": {"max_num_steps": 132},
}

"""
=================================TEST CONFIG=================================

"""

test_cfg = {
    "format_version": 4,
    "model_params": {
        "history_num_frames": 10,
        "history_step_size": 1,
        "history_delta_time": 0.1,
        "future_num_frames": 50,
        "future_step_size": 1,
        "future_delta_time": 0.1,
    },
    "raster_params": {
        "raster_size": [400, 400],
        "pixel_size": [0.1, 0.1],
        "ego_center": [0.25, 0.5],
        "map_type": "py_semantic",
        "satellite_map_key": "aerial_map/aerial_map.png",
        "semantic_map_key": "semantic_map/semantic_map.pb",
        "dataset_meta_key": "meta.json",
        "filter_agents_threshold": 0.5,
    },
    "data_loader_data": {"key": "scenes/test.zarr", "batch_size": 256, "shuffle": False, "num_workers": 0},
}

"""
=================================LYFT PATH ENV SETTINGS=================================

"""

SINGLE_SUB_PATH = (os.path.join(trainer_params["data_path"], "single_mode_sample_submission.csv"),)
MULTI_SUB_PATH = os.path.join(trainer_params["data_path"], "multi_mode_sample_submission.csv")

# Setting env variable for L5KIT ( LYFT framework )
os.environ["L5KIT_DATA_FOLDER"] = trainer_params["data_path"]
data_Manager = LocalDataManager(None)

"""
=================================EXPERIMENT SETTING =================================

"""


def set_experiment(resume_path=None):
    experiment_config = {
        "model": {
            "model_name": Params["net_type"],
            "optimizer": Params["optimizer"],
            "_step_scheduler": False,
            "learning_rate": Params["learning_rate"],
            "batch_size": train_cfg["data_loader_data"]["batch_size"],
            "epochs": Params["EPOCHS"],
        },
        "config": {
            "img_size": train_cfg["raster_params"]["raster_size"][0],
            "pixel_size": train_cfg["raster_params"]["pixel_size"][0],
            "map_type": train_cfg["raster_params"]["map_type"],
            "steps": "train_ALL",
            "_history_num_frames": 10,
            "_history_step_size": 1,
            "_history_delta_time": 0.1,
            "_future_num_frames": 50,
            "_future_step_size": 1,
            "_future_delta_time": 0.1,
        },
    }

    if resume_path is None:
        experiment = Experiment(experiment_config)
    else:
        experiment = Experiment(resume_from=resume_path)
    experiment.register_directory("checkpoints")
    trainer_params["save_path"] = os.path.join(Path(__file__).parent.absolute(), experiment.checkpoints)
    trainer_params["experiment_path"] = os.path.join(
        Path(__file__).parent.absolute(), "experiments", experiment.config.identifier
    )
    trainer_params["experiment"] = experiment


"""
=================================CREATING LOSS FN=================================

"""

# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py # noqa: E501


def pytorch_neg_multi_log_likelihood_batch(gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (
        batch_size,
        future_len,
        num_coords,
    ), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(
        torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))
    ), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(gt: Tensor, pred: Tensor, avails: Tensor) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)


"""
=========================================== CREATING DATALOADERS/DATASETS ===========================================
"""


class LyftImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, data_list):
        super().__init__()
        self.data_folder = data_folder
        self.files = data_list

    def __getitem__(self, index: int):
        return self.obj_load(self.files[index])

    def obj_load(self, name):
        with bz2.BZ2File(f"{self.data_folder}/{name}", "rb") as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.files)


def create_ds_images(cfg: Dict, split_data):
    train_data_dir = os.path.join(
        "cache",
        "pre_{}px__{}__ALL".format(
            train_cfg["raster_params"]["raster_size"][0], int(train_cfg["raster_params"]["pixel_size"][0] * 100)
        ),
    )

    all_files = []

    for filename in os.listdir(train_data_dir):
        all_files.append(filename)

    if split_data:
        train_data, validate_data = train_test_split(all_files, shuffle=True, test_size=0.1)

        valid_dataset = LyftImageDataset(train_data_dir, validate_data)

    else:
        train_data = all_files

    train_dataset = LyftImageDataset(train_data_dir, train_data)

    return train_dataset, valid_dataset


"""
=================================MODEL ETC=================================

"""


class LyftModel(nn.Module):
    def __init__(self, cfg: Dict):
        super().__init__()

        self.backbone = models.resnet.resnet18(pretrained=True, progress=True)

        # print(self.backbone)

        # input channels size to match rasterizer shape
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        input_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            input_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # output_size to (X, y) * number of future states

        self.num_modes = 3
        self.future_len = cfg["model_params"]["future_num_frames"]
        self.output_size = 2 * self.future_len * self.num_modes

        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("head_Flatten", nn.Flatten()),
                    ("head_Linear_1", nn.Linear(in_features=self.backbone.fc.in_features, out_features=4096)),
                    ("head_ELU", nn.ELU()),
                    ("head_Linear_out", nn.Linear(in_features=4096, out_features=self.output_size + self.num_modes)),
                ]
            )
        )

        self.backbone.fc = self.head

    def forward(self, x):
        x_out = self.backbone(x)
        bs, _ = x_out.shape

        pred, confidences = torch.split(x_out, self.output_size, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


"""
=================================PREDICTION METHOD=================================
"""


def make_prediction(prediction_data):
    pred_coords_list = []
    confidences_list = []
    timestamps_list = []
    track_id_list = []

    for ele in prediction_data:
        y_pred, confidences = ele[0]
        data = ele[1]

        pred_coords_list.append(y_pred.cpu().numpy().copy())
        confidences_list.append(confidences.cpu().numpy().copy())
        timestamps_list.append(data["timestamp"].numpy().copy())
        track_id_list.append(data["track_id"].numpy().copy())

    timestamps = np.concatenate(timestamps_list)
    track_ids = np.concatenate(track_id_list)  # noqa: F841
    coords = np.concatenate(pred_coords_list)  # noqa: F841
    confs = np.concatenate(confidences_list)  # noqa: F841

    write_pred_csv(
        os.path.join(trainer_params["experiment_path"], "submission.csv"),
        timestamps=np.concatenate(timestamps),
        track_ids=np.concatenate(agent_ids),  # noqa: F821
        coords=np.concatenate(future_coords_offsets_pd),  # noqa: F821
    )


"""
=================================CREATING CLASS WRAPPER=================================
"""


class LytfTrainingWrapper(TrainerClass):
    def __init__(self, model=None, loss_fn=None):
        super().__init__(model, loss_fn)

    def forward(self, batch_data):
        return self.model(batch_data)

    def train_step(self, batch_data):
        inputs = batch_data["image"].to(device)
        y_pred, confidences = self.model(inputs)

        target_availabilities = batch_data["target_availabilities"].to(device)
        y_true = batch_data["target_positions"].to(device)

        loss = self.loss_fn(y_true, y_pred, confidences, target_availabilities)

        metrics = {
            "loss": loss.item(),
            "nll": pytorch_neg_multi_log_likelihood_batch(y_true, y_pred, confidences, target_availabilities).item(),
        }

        return loss, metrics

    def validation_step(self, batch_data):
        inputs = batch_data["image"].to(device)
        y_pred, confidences = self.model(inputs)

        return (y_pred, confidences)

    def get_optimizer_scheduler(self):
        params = list(self.model.named_parameters())

        def is_head(name):
            return "head" in name

        optimizer_grouped_parameters = [
            {"params": [p for n, p in params if not is_head(n)], "lr": Params["learning_rate"] / 50},
            {"params": [p for n, p in params if is_head(n)], "lr": Params["learning_rate"]},
        ]

        optimizer = Ranger(optimizer_grouped_parameters, lr=Params["learning_rate"])

        scheduler = get_flat_cosine_schedule(
            optimizer=optimizer, num_training_steps=train_cfg["train_params"]["max_num_steps"] * Params["EPOCHS"]
        )

        return optimizer, scheduler


"""
=================================MAIN LOOP=================================

"""


def training():
    set_experiment()

    shutil.copy(__file__, os.path.join(trainer_params["experiment_path"], __file__))
    cfg = train_cfg

    model = LyftModel(cfg).to(device)

    description = "{}_{}_{}_{}_{}_{}".format(
        Params["net_type"],
        Params["optimizer"],
        Params["learning_rate"],
        cfg["data_loader_data"]["batch_size"],
        cfg["raster_params"]["raster_size"][0],
        cfg["raster_params"]["pixel_size"][0],
    )

    trainer_params["description"] = description

    trainer = Trainer(model=LytfTrainingWrapper(model, pytorch_neg_multi_log_likelihood_batch), cfg=trainer_params)

    train_ds, valid_ds = create_ds_images(cfg, split_data=True)

    trainer.fit(
        train_dataset=train_ds,
        batch_size=cfg["data_loader_data"]["batch_size"],
        epochs=Params["EPOCHS"],
        validation_dataset=valid_ds,
        validation_metric="nll",
        steps_per_epoch=40,
    )


def create_predict_dl(cgf):
    return 1


def predict(experiment_dir=None, checkpoint_dir=None):

    assert experiment_dir is not None
    set_experiment(resume_path=experiment_dir)
    cfg = test_cfg

    model = LyftModel(cfg).to(device)

    assert checkpoint_dir is not None
    model_dict = torch.load(args.checkpoint_dir, map_location=device)
    model.load_state_dict(model_dict)

    test_dl = create_predict_dl(cfg)

    trainer = Trainer(model=LytfTrainingWrapper(model, pytorch_neg_multi_log_likelihood_batch))

    prediction_data = trainer.predict(test_dl)

    make_prediction(prediction_data)


def main(args):

    print("raster size ", train_cfg["raster_params"]["raster_size"])
    train_cfg["raster_params"]["raster_size"] = [args.raster_size, args.raster_size]
    print("new raster size ", train_cfg["raster_params"]["raster_size"])

    print("raster size ", train_cfg["raster_params"]["pixel_size"])
    train_cfg["raster_params"]["pixel_size"] = [args.pixel_size, args.pixel_size]
    print("new raster size ", train_cfg["raster_params"]["pixel_size"])

    print("batch_size ", train_cfg["data_loader_data"]["batch_size"])
    train_cfg["data_loader_data"]["batch_size"] = args.batch_size
    print("new batch_size ", train_cfg["data_loader_data"]["batch_size"])

    Params["EPOCHS"] = args.epochs

    if args.training_mode:
        training()
    else:
        predict(experiment_dir=args.experiment_dir, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixel_size", type=float, required=True)
    parser.add_argument("--raster_size", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=False, default=None)
    parser.add_argument("--experiment_dir", type=str, required=False, default=None)
    parser.add_argument("--epochs", type=int, required=False, default=1)
    parser.add_argument("--training_mode", type=bool, required=False, default=True)
    args = parser.parse_args()
    main(args)
