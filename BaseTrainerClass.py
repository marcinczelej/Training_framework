from abc import ABC, abstractmethod
import torch

class TrainerClass(ABC):
    @abstractmethod
    def __init__(self, model:torch.nn.Module=None, loss_fn=None):
        
        assert(loss_fn!=None)
        assert(model!=None)
        self.model = model
        self.loss_fn = loss_fn
    
    @abstractmethod
    def train_step(self, batch_data):
        pass
    
    @abstractmethod
    def validation_step(self, batch_data):
        pass
    
    @abstractmethod
    def get_optimizer_scheduler(self):
        pass
    
