from simple_framework.callbacks.BaseCallbackClass import CallbackBase
import torch
import os
import logging

from pathlib import Path


class CheckpointCallback(CallbackBase):
    def __init__(
        self,
        save_dir: str = "",
        frequency: int = 1,
        type: str = "step",
        phase: str = "train",
        mode: str = "min",
        metric: str = "loss",
        prefix: str = "",
        weights_only: bool = False,
        save_last: bool = True,
        save_best: bool = True,
    ):
        self.save_dir = save_dir
        self.frequency = frequency
        self.type = type
        self.phase = phase
        self.mode = mode
        self.metric = metric
        self.prefix = prefix
        self.weights_only = weights_only
        self.save_last = save_last
        self.save_best = save_best

        if mode == "min":
            self.best_val = 100000000
        else:
            self.best_val = -100000000

    def on_train_epoch_end(self, trainer, module):
        if self.type != "epoch" or self.phase != "train" or (trainer.current_epoch + 1) % self.frequency != 0:
            return

        self.__save_routine(trainer, module)

    def on_validation_epoch_end(self, trainer, module):
        if self.type != "epoch" or self.phase != "validation" or (trainer.current_epoch + 1) % self.frequency != 0:
            return

        self.__save_routine(trainer, module)

    def on_train_step_end(self, trainer, module):
        if self.type != "step" or self.phase != "train" or (trainer.current_step + 1) % self.frequency != 0:
            return

        self.__save_routine(trainer, module)

    """
    ======================= MAIN SAVE ROUTINE =======================
    """

    def __save_routine(self, trainer, module):

        epoch_dir = os.path.join(self.save_dir, f"epoch_{trainer.current_epoch}")
        Path(epoch_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_name = (
            f"{epoch_dir}/Train_epoch_{self.prefix}_{trainer.current_epoch}_step_{trainer.current_step}_checkpoint"
        )

        self.__save_checkpoint(trainer=trainer, module=module, name=checkpoint_name)

        self.__should_save_best(trainer, module)
        self.__should_save_last(trainer, module)

    """
    ======================= SAVE ALL/WEIGHTS_ONLY =======================
    """

    def __save_weights_only(self, module, name):

        checkpoint_name = name + ".pth"
        torch.save(module.state_dict(), checkpoint_name)
        logging.info(f"saving weights only to {checkpoint_name}")

    def __save_all_data(self, trainer, module, name):
        checkpoint_name = name + ".bin"

        module.eval()

        save_dict = {
            "model_state_dict": module.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "epoch": trainer.current_epoch,
            "best_metric": self.best_val,
        }

        if trainer.scheduler is not None:
            save_dict["scheduler_state_dict"] = trainer.scheduler.state_dict()

        print(checkpoint_name)

        torch.save(save_dict, checkpoint_name)

    """
    ======================= SAVE BEST/LAST =======================
    """

    def __should_save_best(self, trainer, module):
        if self.save_best is False or self.mode == "auto":
            return

        if (self.mode == "min" and trainer.metric_container[self.metric].avg < self.best_val) or (
            self.mode == "max" and trainer.metric_container[self.metric].avg > self.best_val
        ):

            save_path = os.path.join(self.save_dir, "best_checkpoint")
            self.__save_checkpoint(trainer, module, save_path)
            self.best_val = trainer.metric_container[self.metric].avg

    def __should_save_last(self, trainer, module):
        if self.save_last:
            save_path = os.path.join(self.save_dir, "last_checkpoint")
            self.__save_checkpoint(trainer, module, save_path)

    """
    ======================= SAVING CHOOSER - ALL OR ONLY MODEL =======================
    """

    def __save_checkpoint(self, trainer, module, name):
        if self.weights_only:
            self.__save_weights_only(module=module, name=name)
        else:
            self.__save_all_data(trainer=trainer, module=module, name=name)
