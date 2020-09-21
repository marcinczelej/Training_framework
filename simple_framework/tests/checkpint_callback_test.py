import os
import pytest

from pathlib import Path

from typing import List

import shutil

from simple_framework.tests.Tests_base import getSimpleModel, getSimpleDataset, getParameters
from simple_framework.trainer.trainer import Trainer
from simple_framework.callbacks.CheckpointCallback import CheckpointCallback


@pytest.mark.parametrize(
    "backbone,frequency, callback_type, expected",
    [
        ("Horovod", 1, "step", 20),
        ("Horovod", 1, "epoch", 2),
        ("Horovod", 2, "step", 10),
        ("Horovod", 2, "epoch", 1),
        ("Horovod", 10, "step", 2),
        ("Horovod", 10, "epoch", 0),
        ("Simple", 1, "step", 20),
        ("Simple", 1, "epoch", 2),
        ("Simple", 2, "step", 10),
        ("Simple", 2, "epoch", 1),
        ("Simple", 10, "step", 2),
        ("Simple", 10, "epoch", 0),
    ],
)
def test_checkpoint_callback(backbone: str, frequency: int, callback_type: str, expected: int):
    """
    2 epochs, 10 steps each
    """
    tmp_dir = "saved_files"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    model = getSimpleModel()
    params = getParameters(backbone)
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

    shutil.rmtree(tmp_dir)


# ---------------------------- CsvCallback Test ----------------------------
