from abc import ABC, abstractmethod
import torch


class SimpleFrameworkWrapper(ABC, torch.nn.Module):
    @abstractmethod
    def __init__(self, model: torch.nn.Module = None, loss_fn=None):
        super().__init__()
        assert loss_fn is not None
        assert model is not None
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    @abstractmethod
    def train_step(self, batch_data):
        pass

    @abstractmethod
    def validation_step(self, batch_data):
        pass

    @abstractmethod
    def get_optimizer_scheduler(self):
        pass
