from simple_framework.trainer.BaseTrainerClass import SimpleFrameworkWrapper

from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class BaseTestModel(SimpleFrameworkWrapper):
    def __init__(self, model=None, loss_fn=None):
        super().__init__(model, loss_fn)

    def forward(self, batch_data):
        return self.model(batch_data)

    def train_step(self, batch_data):
        data, y_true = batch_data
        y_pred = self.model(data)

        loss = self.loss_fn(y_pred, y_true, size_average=False)

        # get the index of the max log-probability
        pred = y_pred.data.max(1, keepdim=True)[1]
        accuracy = pred.eq(y_true.data.view_as(pred)).cpu().float().sum()

        metrics = {
            "loss": loss.item(),
            "acc": accuracy,
        }

        return loss, metrics

    def validation_step(self, batch_data):
        pass

    def get_optimizer_scheduler(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.5)

        return optimizer, None


Path("simple_f").mkdir(parents=True, exist_ok=True)


def getSimpleModel():
    return BaseTestModel(Net(), F.nll_loss)


def getSimpleDataset():
    return datasets.MNIST(
        "data-%d",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
    )


def getParameters(backend):
    return {
        "save_path": "/simple_f",
        "experiment_path": "simple_f",
        "description": "MNIST",
        "step_scheduler": False,
        "validation_scheduler": False,
        "backbone": backend,
    }
