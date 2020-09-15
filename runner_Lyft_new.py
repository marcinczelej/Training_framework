import os
import sys
import shutil
import argparse
import math
import logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from torch.utils.tensorboard import SummaryWriter

from pathlib import Path

from typing import Dict

from tqdm import tqdm

from ranger import Ranger

import bz2
import pickle

import torchvision.models as models

from mag.experiment import Experiment

from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv

from sklearn.model_selection import train_test_split
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

print(torch.cuda.is_available())

writer = SummaryWriter()

"""
=================================PARAMS=================================

"""

Params = {
    "data_path" : "data",
    "save_path" : "checkpoints",
    "checkpoint_path" : "checkpoints/best_model_checkpoint.pth",
    "experiment_path": "data",
    "debug" : False,
    "step_scheduler": True, 
    "epoch_scheduler": False,
    "EPOCHS": 1, 
    "training_mode": True,
    "experiment": None,
    "optimizer":"Ranger_flat_cosine",
    "learning_rate": 1e-4,
    "net_type": "Resnet18",
    "apex_opt_level": 1,
    "use_apex":False,
}

"""
Configuration for LYFT  datasets/framework

history_num_frames - how many frames to take from history at once
history_step_size - what interval to have 

for example:

    frame_index == 10
    history_num_frames == 4
    history_step_size == 2

will return:
    10, 8, 6, 4
    
    
"""

map_types = {
    "semantic": "py_semantic", 
    "satelite": "py_satellite",
}


"""
=================================TRAIN CONFIG=================================

"""

train_cfg = {
    'format_version': 4,
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [400, 400],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    'data_loader_data': {
        'key': 'scenes/train.zarr',
        'batch_size': 4,
        'shuffle': True,
        'num_workers': 0
    },
    
    'train_params': {
        'max_num_steps': 100 if Params["debug"] else 132,
        'checkpoint_every_n_steps': 2000,
        
        # 'eval_every_n_steps': -1
    }
}

"""
=================================TEST CONFIG=================================

"""

test_cfg = {
    'format_version': 4,
    'model_params': {
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1
    },
    
    'raster_params': {
        'raster_size': [400, 400],
        'pixel_size': [0.1, 0.1],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },
    
    'data_loader_data': {
        'key': 'scenes/test.zarr',
        'batch_size': 256,
        'shuffle': False,
        'num_workers': 0
    }

}

"""
=================================LYFT PATH ENV SETTINGS=================================

"""

SINGLE_SUB_PATH = os.path.join(Params["data_path"], "single_mode_sample_submission.csv"),
MULTI_SUB_PATH = os.path.join(Params["data_path"], "multi_mode_sample_submission.csv")

# Setting env variable for L5KIT ( LYFT framework )
os.environ["L5KIT_DATA_FOLDER"] = Params["data_path"]
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
            "epochs": Params["EPOCHS"]
        },
        "config": {
            "img_size": train_cfg["raster_params"]["raster_size"][0],
            "pixel_size": train_cfg["raster_params"]["pixel_size"][0], 
            "map_type": train_cfg["raster_params"]["map_type"], 
            "steps": "train_ALL", 
            '_history_num_frames': 10,
            '_history_step_size': 1,
            '_history_delta_time': 0.1,
            '_future_num_frames': 50,
            '_future_step_size': 1,
            '_future_delta_time': 0.1,
        }
    }
    
    if resume_path == None:
        experiment = Experiment(experiment_config)
    else:
        experiment = Experiment(resume_from=resume_path)
    experiment.register_directory("checkpoints")
    Params["save_path"] = experiment.checkpoints
    Params["experiment_path"] = os.path.join("experiments", experiment.config.identifier)
    
    logging_dir = os.path.join(Path(__file__).parent.absolute(), Params["experiment_path"], 'info.log')
    print("logging to ", logging_dir)
    
    logging.StreamHandler(stream=None)
    logger = logging.getLogger()
    
    fhandler = logging.FileHandler(filename=logging_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(process)d -  %(message)s')
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
        
    logging.info("experiment.config.identifier = ", experiment.config.identifier)
    Params["experiment"] = experiment
    
"""
=================================CREATING LOSS FN=================================

"""  

# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
import numpy as np

import torch
from torch import Tensor


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
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

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
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


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
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

        logging.info(f"Dataset files len {len(self.files)}")

    def __getitem__(self, index: int):
        return self.obj_load(self.files[index])

    def obj_load(self, name):
        with bz2.BZ2File(f'{self.data_folder}/{name}', 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return len(self.files)
    
def create_dl_images(cfg: Dict, split_data):
    train_data_dir = os.path.join('cache', "pre_{}px__{}__ALL".format(train_cfg["raster_params"]["raster_size"][0], int(train_cfg["raster_params"]["pixel_size"][0]*100)))
    
    all_files = []
    
    for filename in os.listdir(train_data_dir):
        all_files.append(filename)
    
    valid_dataloader = None
    
    if split_data:
        logging.info("going to split data...")
        train_data, validate_data = train_test_split(all_files, shuffle=True, test_size=0.1)
        
        print("validation_dataset len ", len(validate_data))
        print("batch_size ", cfg["data_loader_data"]["batch_size"])
        
        valid_dataset = LyftImageDataset(train_data_dir, validate_data)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                  shuffle=cfg["data_loader_data"]["shuffle"], 
                                                  batch_size=cfg["data_loader_data"]["batch_size"], 
                                                  num_workers=cfg["data_loader_data"]["num_workers"])

    else:
        train_data = all_files
    
    
    train_dataset = LyftImageDataset(train_data_dir, train_data)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  shuffle=cfg["data_loader_data"]["shuffle"], 
                                                  batch_size=cfg["data_loader_data"]["batch_size"], 
                                                  num_workers=cfg["data_loader_data"]["num_workers"])

    logging.info(f"train dataloader len {len(train_dataloader)}")
    
    if split_data:
        logging.info(f"valid dataloader len {len(valid_dataloader)}")
    logging.info(f"all files len {len(all_files)}")

    return train_dataloader, valid_dataloader

"""
=================================MODEL ETC=================================

"""

class LyftModel(nn.Module):
    def __init__(self, cfg: Dict):
        super(LyftModel, self).__init__()
        
        self.backbone = models.resnet.resnet18(pretrained=True, 
                                               progress=True,
                                              )
        
        #print(self.backbone)
        
        # input channels size to match rasterizer shape
        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1)*2
        input_channels = 3 + num_history_channels
        
        self.backbone.conv1 = nn.Conv2d(input_channels, 
                                       self.backbone.conv1.out_channels, 
                                       kernel_size=self.backbone.conv1.kernel_size, 
                                       stride=self.backbone.conv1.stride,
                                       padding=self.backbone.conv1.padding,
                                       bias=False,
                                       )
        
        # output_size to (X, y) * number of future states
        
        self.num_modes=3
        self.future_len = cfg["model_params"]["future_num_frames"]
        self.output_size = 2 * self.future_len *self.num_modes
        
        self.head = nn.Sequential(OrderedDict([
                            ('head_Flatten', nn.Flatten()),
                            ('head_Linear_1', nn.Linear(in_features=self.backbone.fc.in_features, out_features=4096)),
                            ('head_ELU', nn.ELU()), 
                            ('head_Linear_out', nn.Linear(in_features=4096, out_features=self.output_size+self.num_modes)),
        ]))
        
        self.backbone.fc = self.head

    def forward(self, x):
        x_out = self.backbone(x)
        bs, _ = x_out.shape


        pred, confidences = torch.split(x_out, self.output_size, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

def get_flat_cosine_schedule(optimizer, num_training_steps, percentege_of_const=0.7, num_cycles=0.5, last_epoch=-1):
    """
        Before percentage_of_const get constant lr,
        then do cosine decay till ends
    """
    logging.info("get_flat_cosine_schedule: num_training_steps = {}" .format(num_training_steps))
    #print("int(num_training_steps*percentege_of_const) = ", int(num_training_steps*percentege_of_const))
    
    def lr_lambda(current_step):
        if current_step < int(num_training_steps*percentege_of_const):
            #print(current_step, " returning 1")
            return 1.0
        progress = float(current_step - num_training_steps*percentege_of_const) / float(max(1, num_training_steps - num_training_steps*percentege_of_const))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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
        self.avg = self.sum / self.count# / (self.count*Parameters.batch_accumulation)

"""
=================================CREATING CLASS WRAPPER=================================
"""        

class LytfTrainingWrapper(nn.Module):
    def __init__(self, model, loss_fn=None):
        super().__init__()
        
        assert(loss_fn != None)
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, batch_data):
        
        inputs = batch_data["image"].to(device)
        target_availabilities = batch_data["target_availabilities"].to(device)
        y_true = batch_data["target_positions"].to(device)
        
        y_pred, confidences = self.model(inputs)
        loss = self.loss_fn(y_true, y_pred, confidences, target_availabilities)
        
        metrics = {
            "loss": loss.item(),
            "nll": pytorch_neg_multi_log_likelihood_batch(y_true, y_pred, confidences, target_availabilities).item(),
        }
        
        return loss, metrics

"""
=================================CHECKPOINTER=================================
"""


    
"""
=================================CREATING TRAINER CLASS=================================
"""

class Trainer():
    def __init__(self, optimizer, model, scheduler, cfg, prefix):
        self.optimizer = optimizer
        self.model = model
        self.scheduler = scheduler
        self.cfg = cfg
        self.prefix = prefix
        self.metric=None
        self.loss_fn=None
        self.best_loss = 10000000
        self.best_validation_metric = 10000000
        self.epoch = 0
        self.metric_container = {}
        
        self.settings = {
            "batch_size":None,
            "epochs":1,
            "steps_per_epoch":None,
            "validation_steps":None,
            "validation_batch_size":None,
            "validation_freq":1,
            "checkpoint_every_n_steps":None,
        }

        if Params["use_apex"]:
            opt_level = Params["apex_opt_level"]
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)

        self.train_columns = ["epoch", "step", "current_loss"]
        self.validate_columns = ["epoch"]
            
        self.train_df = None
        self.valid_df = pd.DataFrame(columns=["epoch", "loss", "avg_metric"])

    def set_model(self, model):
        self.model = model
        
    def __get_learning_rate(self):
        lr=[]
        for param_group in self.optimizer.param_groups:
           lr +=[ param_group['lr'] ]
        return lr
        
    def __train_one_epoch(self, dataloader):
        epochs = self.settings["epochs"]
        self.model.train()
        torch.set_grad_enabled(True)
            
        dl_iter = iter(dataloader)
            
        pbar = tqdm(range(self.settings["steps_per_epoch"]), dynamic_ncols=True)

        for step in pbar:
            try:
                data = next(dl_iter)
            except StopIteration:
                tr_it = iter(dataloader)
                data = next(dl_iter)
            
            tensorflow_step = step + self.epoch*self.settings["steps_per_epoch"]
            self.optimizer.zero_grad()

            loss, metrics = self.model(data)
            
            # creating metrics container first time we see it
            if not self.metric_container:
                for i, (key, val) in enumerate(metrics.items()):
                    self.train_columns.append(key)
                    self.metric_container[key] = AverageMeter()
                    logging.info("adding metric {}" .format(key))
                self.train_columns.append("learning_rate")
                self.train_columns.append("saving_best")
                self.train_df = pd.DataFrame(columns=self.train_columns)
                                     
            # filling metrics/loss
            for i, (key, val) in enumerate(metrics.items()):
                self.metric_container[key].update(val, self.settings["batch_size"])

            if Params["use_apex"]:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()

            if Params["step_scheduler"]:
                self.scheduler.step() 

            # add to average meter for loss
            #train_sum_loss.update(loss.detach().item(), self.cfg["data_loader_data"]["batch_size"])

            # writer to tensorboard
            writer.add_scalar('Loss/train_{}' .format(self.prefix), 
                              self.metric_container["loss"].avg, tensorflow_step)
            
            writer.add_scalar('Loss/all_train_losses', self.metric_container["loss"].avg, tensorflow_step)
            
            # adding all metrics to tensorboard
            for i, (key, val) in enumerate(self.metric_container.items()):
                if key == "loss":
                    continue
                writer.add_scalar('Metric/{}_{}' .format(key, self.prefix), val.avg, tensorflow_step)
                writer.add_scalar('Metric/{}_all_runs' .format(key), val.avg, tensorflow_step)
            
            # adding all learning rates to tensorboard
            for idx, lr_value in enumerate(self.__get_learning_rate()):
                writer.add_scalar("Learning_rates/lr_{}_{}" .format(idx, self.prefix), lr_value, tensorflow_step)

            #set pbar description
            current_loss = self.metric_container["loss"].current
            avg_loss = self.metric_container["loss"].avg
            pbar.set_description(f"TRAIN epoch {self.epoch+1}/{epochs} idx {step} current loss {current_loss}, avg loss {avg_loss}")

            #Saving interval - optional
            if self.settings['checkpoint_every_n_steps'] != None and (step+1) %  self.settings['checkpoint_every_n_steps'] == 0:
                    save_dir = os.path.join(Params["save_path"], f"epoch_{self.epoch}")
                    Path(save_dir).mkdir(parents=True, exist_ok=True)
                    save_file = os.path.join(save_dir, f"epoch_{self.epoch}_step_{step}_avg_loss_{avg_loss}.bin")
                    self.save(save_file)

            #Saving best loss and registering best loss
            saving_best=False
            if self.best_loss > self.metric_container["loss"].avg:
                saving_best=True
                logging.info("saving best model with loss {}" .format(self.metric_container["loss"].avg))
                self.best_loss = self.metric_container["loss"].avg
                torch.save(self.model.state_dict(), os.path.join(Params["save_path"], "best_model_checkpoint.pth"))
                Params["experiment"].register_result("best_loss", self.metric_container["loss"].avg)

            #registering last loss
            Params["experiment"].register_result("last_loss", self.metric_container["loss"].avg)
            
            # adding given step data to csv file
            new_row = [self.epoch, step, loss.detach().item()] +  [metric.avg for metric in self.metric_container.values()] + [self.__get_learning_rate(), saving_best]
            
            new_series = pd.Series(new_row, index=self.train_df.columns)

            self.train_df = self.train_df.append(new_series, ignore_index=True)
            self.train_df.to_csv(os.path.join(Params["experiment_path"], "train_logs.csv"), index=False)
        
        # Saving last checkpoint in epoch
        save_file = os.path.join(Params["save_path"], "last_checkpoint.bin")
        self.save(save_file)
        
    def __validation_step(self, dataloader, validation_metric):
        epochs = self.settings["epochs"]
        validation_sum_loss = AverageMeter()
        val_metric = AverageMeter()
        self.model.eval()
        
        pred_coords_list = []
        confidences_list = []
        timestamps_list = []
        track_id_list = []
        
        with torch.no_grad():
            print("len dataloader ", len(dataloader))
            dl_iter = iter(dataloader)
            
            pbar = tqdm(range(self.settings["validation_steps"]), dynamic_ncols=True)

            for step in pbar:
                try:
                    data = next(dl_iter)
                except StopIteration:
                    tr_it = iter(dataloader)
                    data = next(dl_iter)
                
                loss, metrics = self.model(data)

                # add to average meter for loss
                validation_sum_loss.update(loss.item(), self.settings["validation_batch_size"])
                
                # validation metric update
                val_metric.update(metrics[validation_metric], self.settings["validation_batch_size"])

                #set pbar description
                pbar.set_description(f"VALIDATION epoch {self.epoch+1}/{epochs} step {step} current loss {validation_sum_loss.current}, avg loss {validation_sum_loss.avg}, validation_metric {val_metric.avg}")
            
            # adding given epoch data to csv file
            new_row = [self.epoch, validation_sum_loss.avg, val_metric.avg]
            
            new_series = pd.Series(new_row, index=self.valid_df.columns)

            self.valid_df = self.valid_df.append(new_series, ignore_index=True)
            self.valid_df.to_csv(os.path.join(Params["experiment_path"], "validation_logs.csv"), index=False)
            logging.info("Validation result metric for epoch {} = {}" .format(self.epoch, val_metric.avg))
                
            if self.best_validation_metric > val_metric.avg:
                file_name = "best_model_validation_{}.pth" .format(self.prefix)
                
                logging.info("saving best model with epoch {} validation metric {} as {}" .format(self.epoch, val_metric.avg, file_name))
                self.best_validation_metric = val_metric.avg
                torch.save(self.model.state_dict(), os.path.join(Params["save_path"], file_name))
                Params["experiment"].register_result("best_validation_loss", validation_sum_loss.avg)
                Params["experiment"].register_result("best_validation_metric", val_metric.avg)
                
                                 
    
    def __predict_outputs(self, data_loader):
        self.model.eval()

        pred_coords_list = []
        confidences_list = []
        timestamps_list = []
        track_id_list = []

        with torch.no_grad():
            pbar = tqdm(dataloader, 
                    total=len(dataloader), 
                    dynamic_ncols=True
                    )
            for data in dataiter:
                image = data["image"].to(device)

                y_pred, confidences = self.model(inputs)

                pred_coords_list.append(y_pred.cpu().numpy().copy())
                confidences_list.append(confidences.cpu().numpy().copy())
                timestamps_list.append(data["timestamp"].numpy().copy())
                track_id_list.append(data["track_id"].numpy().copy())
                
        timestamps = np.concatenate(timestamps_list)
        track_ids = np.concatenate(track_id_list)
        coords = np.concatenate(pred_coords_list)
        confs = np.concatenate(confidences_list)
        
        write_pred_csv(os.path.join(Params["experiment_path"], 'submission.csv'),
               timestamps=np.concatenate(timestamps),
               track_ids=np.concatenate(agent_ids),
               coords=np.concatenate(future_coords_offsets_pd))

    
    def save(self, path):
        logging.info("saving checkpoint to {}" .format(path))
        self.model.eval()
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'best_loss': self.best_loss,
        }
        
        if Params["step_scheduler"]:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if Params["use_apex"]:
            save_dict["amp"] =  amp.state_dict()
        
        torch.save(save_dict, path)
    
    def load(self, path):
        logging.info("loading checkpoint from {}" .format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epoch = checkpoint['epoch']+1
        self.best_loss = checkpoint['best_loss']
        
        if Params["step_scheduler"]:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if Params["use_apex"]:
            amp.load_state_dict(checkpoint['amp'])
            
    def fit(self, train_dataloader=None, batch_size=None, epochs=1, validation_dataloader=None, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, checkpoint_every_n_steps=None, validation_metric='loss'):
        """
        Inputs:
            :param Dataloader train_dataloader: Dataloader for model.
            :param int batch_size: batch size that should be used during training/validation
            :param int epochs: NUmber of epochs to train
            :param DataLoader validation_dataloader: Dataloader for validation data
            :param int steps_per_epoch: training steps that should be performed each epoch. If not specified whole training set would be used
            :param int validation_steps: validation steps that should be  performed each validation phase. If not specified, whole validation set would be used
            :param int validation_batch_size: Batch size for validation step. If not set, training batch size would be used
            :param int validation_freq:After how many epochs validation should be performed
            :param int checkpoint_every_n_steps: should be set if we want to save model in given timesteps interval
            :param str validation_metric: metric that should be checkd after validation to see if result is better.
        """
        
        assert(batch_size!=None)
        assert(train_dataloader!=None)
        self.settings["batch_size"]=batch_size
        self.settings["epochs"]=epochs
        
        if steps_per_epoch==None:
            self.settings["steps_per_epoch"]=len(train_dataloader)
        else:
            self.settings["steps_per_epoch"]=steps_per_epoch
        
        if validation_dataloader!=None:
            self.settings["validation_steps"]=len(validation_dataloader)
        else:
            self.settings["validation_steps"]=validation_steps
        
        if validation_batch_size==None:
            self.settings["validation_batch_size"]=batch_size
        else:
            self.settings["validation_batch_size"]=validation_batch_size
        
        
        self.settings["validation_freq"]=validation_freq
        self.settings["checkpoint_every_n_steps"]=checkpoint_every_n_steps
        
        assert(self.model != None)
        assert(self.optimizer !=None)
        
        logging.info("Training settings:")
        for _, (key, val) in enumerate(self.settings.items()):
            logging.info("{} : {}" .format(key, val))
        
        for epoch in range(self.settings["epochs"]):
            logging.info("starting epoch {}/{} training step" .format(epoch+1, self.settings["epochs"]))
            self.epoch = epoch
            self.__train_one_epoch(train_dataloader)
        
            if validation_dataloader != None and (epoch+1)% self.settings["validation_freq"]==0:
                self.__validation_step(validation_dataloader, 
                                       validation_metric=validation_metric)
            
    def predict(self, dataloader):
        
        assert(self.model != None)
        
        self.__predict_outputs(dataloader)

"""
=================================MAIN LOOP=================================

"""

def main(args):
    
    print("raster size ",train_cfg["raster_params"]["raster_size"])
    train_cfg["raster_params"]["raster_size"] = [args.raster_size, args.raster_size]
    print("new raster size ",train_cfg["raster_params"]["raster_size"])
    
    print("raster size ",train_cfg["raster_params"]["pixel_size"])
    train_cfg["raster_params"]["pixel_size"] = [args.pixel_size, args.pixel_size]
    print("new raster size ",train_cfg["raster_params"]["pixel_size"])
    
    print("batch_size ",train_cfg["data_loader_data"]["batch_size"])
    train_cfg["data_loader_data"]["batch_size"] = args.batch_size
    print("new batch_size ",train_cfg["data_loader_data"]["batch_size"])
    
    Params["EPOCHS"] = args.epochs
    
    if Params["training_mode"]:
        set_experiment()
        
        logging.info("Training mode")
        shutil.copy(__file__, os.path.join(Params["experiment_path"], __file__))
        cfg = train_cfg
    else:
        set_experiment(resume_path=args.experiment_dir)

        logging.info("Testing mode")
        cfg = test_cfg
    
    model = LyftModel(cfg).to(device)
    model = nn.DataParallel(model)
    
    optimizer = None
    scheduler = None

    if Params["training_mode"] == False:
        logging.info("Testing mode loading model")
        assert(args.checkpoint_dir != None)
        model_dict = torch.load(args.checkpoint_dir, map_location = device)
        model.load_state_dict(model_dict)
    else:
        logging.info("Training mode creating")
        
        #ResNet
        params = list(model.named_parameters())
        
        def is_head(name):
            return "head" in name
        
        optimizer_grouped_parameters = [
        {"params": [p for n, p in params if not is_head(n)], "lr": Params["learning_rate"]/50},
        {"params": [p for n, p in params if is_head(n)], "lr": Params["learning_rate"]},
        ]
        
        optimizer = Ranger(optimizer_grouped_parameters, lr=Params["learning_rate"])
        scheduler = get_flat_cosine_schedule(optimizer=optimizer, 
                                             num_training_steps=cfg["train_params"]["max_num_steps"]*Params["EPOCHS"])
    
    prefix = '{}_{}_{}_{}_{}_{}' .format(Params["net_type"], 
                                         Params["optimizer"], 
                                         Params["learning_rate"], 
                                         cfg["data_loader_data"]["batch_size"], 
                                         cfg["raster_params"]["raster_size"][0], 
                                         cfg["raster_params"]["pixel_size"][0])
    
    trainer = Trainer(optimizer=optimizer, 
                      model=LytfTrainingWrapper(model, pytorch_neg_multi_log_likelihood_batch), 
                      scheduler=scheduler, 
                      cfg=cfg, 
                      prefix=prefix)
    
    train_dl, valid_dl = create_dl_images(cfg, 
                                          split_data=True)
    
    if Params["training_mode"]:
        logging.info("Starting training...")
        print("Starting training...")
        
        trainer.fit(train_dataloader=train_dl, 
                    batch_size=cfg["data_loader_data"]["batch_size"], 
                    epochs=Params["EPOCHS"],
                    validation_dataloader=valid_dl, 
                    validation_metric='nll')
    else:
        logging.info("Starting prediction...")
        print("Starting prediction...")
        trainer.predict()
        

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument('--pixel_size', type=float, required=True)
    parser.add_argument('--raster_size', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=False, default=None)
    parser.add_argument('--experiment_dir', type=str, required=False, default=None)
    parser.add_argument('--epochs', type=int, required=False, default=1)
    args = parser.parse_args()
    main(args)
