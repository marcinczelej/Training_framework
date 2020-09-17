__version__ = "0.0.1.3"

from simple_framework.trainer import Trainer, TrainerClass

from simple_framework.utilities.schedulers import get_flat_cosine_schedule
from simple_framework.utilities.metrics import AverageMeter
from simple_framework.utilities.checkpoint_saver import Checkpoint_saver
from simple_framework.backends import SimpleBackend, HorovodBackend


__all__ = [
    "Trainer",
    "get_flat_cosine_schedule",
    "AverageMeter",
    "Checkpoint_saver",
    "SimpleBackend",
    "HorovodBackend",
]

# for compatibility with namespace packages
__import__("pkg_resources").declare_namespace(__name__)
