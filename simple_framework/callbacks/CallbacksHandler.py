from typing import List

from simple_framework.callbacks.CheckpointCallback import CheckpointCallback


class CallbacksHandler:
    def __init__(self, callbacks: List):
        self.callbacks = callbacks

    def handle(self, trainer, module, phase):
        if self.callbacks is None:
            return

        for callback in self.callbacks:
            method = None
            try:
                method = getattr(callback, phase)
            except Exception as e:
                print(e)
                continue
            method(trainer, module)
