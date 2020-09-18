import os
import pytest

from pathlib import Path

import shutil

from simple_framework.tests.Tests_base import getSimpleModel, getSimpleDataset, getParameters
from simple_framework.trainer.trainer import Trainer
from simple_framework.callbacks.CheckpointCallback import CheckpointCallback

"""
2 epochs, 10 steps each
"""


@pytest.mark.parametrize(
    "frequency, callback_type, expected",
    [(1, "step", 20), (1, "epoch", 2), (2, "step", 10), (2, "epoch", 1), (10, "step", 2), (10, "epoch", 0)],
)
def test_checkpoint_callback_train(frequency, callback_type, expected):
    """
    Training phase testing
    """
    tmp_dir = "saved_files"
    shutil.rmtree(tmp_dir)

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    model = getSimpleModel()
    params = getParameters()
    dataset = getSimpleDataset()

    trainer = Trainer(model=model, cfg=params)

    checkpoint_callback = CheckpointCallback(
        save_dir=tmp_dir, frequency=frequency, type=callback_type, save_last=False, save_best=False
    )

    trainer.fit(
        train_dataset=dataset,
        batch_size=1,
        epochs=2,
        validation_dataset=None,
        validation_metric="acc",
        steps_per_epoch=10,
        callbacks=[checkpoint_callback],
    )

    file_number = sum([len(files) for r, d, files in os.walk(tmp_dir)])

    assert file_number == expected
