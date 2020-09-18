__version__ = "0.0.1.7"

from simple_framework.trainer import Trainer, SimpleFrameworkWrapper

from simple_framework.utilities.schedulers import get_flat_cosine_schedule
from simple_framework.utilities.metrics import AverageMeter
from simple_framework.backends.horovod_backend import HorovodBackend
from simple_framework.backends.simple_backend import SimpleBackend
from simple_framework.callbacks.BaseCallbackClass import CallbackBase
from simple_framework.callbacks.CheckpointCallback import CheckpointCallback

__all__ = [
    "Trainer",
    "get_flat_cosine_schedule",
    "AverageMeter",
    "Checkpoint_saver",
    "SimpleBackend",
    "HorovodBackend",
    "CheckpointCallback",
    "CallbackBase",
]

print(__name__)

# for compatibility with namespace packages
__import__("pkg_resources").declare_namespace(__name__)
