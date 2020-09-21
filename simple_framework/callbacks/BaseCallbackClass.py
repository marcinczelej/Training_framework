from abc import ABC
from simple_framework.trainer.BaseTrainerClass import SimpleFrameworkWrapper
from simple_framework.backends.BaseBackendClass import BackendBase


class CallbackBase(ABC):
    def on_setup(self, trainer: BackendBase, module: SimpleFrameworkWrapper):
        pass

    def on_train_epoch_start(self, trainer, module):
        pass

    def on_train_epoch_end(self, trainer, module):
        pass

    def on_train_step_start(self, trainer, module):
        pass

    def on_train_step_end(self, trainer, module):
        pass

    def on_validation_epoch_start(self, trainer, module):
        pass

    def on_validation_epoch_end(self, trainer, module):
        pass

    def on_validation_step_start(self, trainer, module):
        pass

    def on_validation_step_end(self, trainer, module):
        pass
