import os
import pytest
import pandas as pd

import collections

from pathlib import Path

from typing import List

import shutil

from simple_framework.tests.Tests_base import getSimpleModel, getSimpleDataset, getParameters
from simple_framework.trainer.trainer import Trainer
from simple_framework.callbacks.CsvCallback import CsvCallback


@pytest.mark.parametrize(
    "backbone, save_interval, save_training, save_validation, expected_file_names, expected_csv_rows_nr",
    [
        ("Horovod", "step", True, False, ["train.csv"], [20]),
        ("Horovod", "step", False, True, ["validation.csv"], [10]),
        ("Horovod", "step", True, True, ["train.csv", "validation.csv"], [20, 10]),
        ("Horovod", "step", False, False, [], []),
        ("Horovod", "epoch", True, False, ["train.csv"], [2]),
        ("Horovod", "epoch", False, True, ["validation.csv"], [2]),
        ("Horovod", "epoch", True, True, ["train.csv", "validation.csv"], [2, 2]),
        ("Horovod", "epoch", False, False, [], []),
        ("Simple", "step", True, False, ["train.csv"], [20]),
        ("Simple", "step", False, True, ["validation.csv"], [10]),
        ("Simple", "step", True, True, ["train.csv", "validation.csv"], [20, 10]),
        ("Simple", "step", False, False, [], []),
        ("Simple", "epoch", True, False, ["train.csv"], [2]),
        ("Simple", "epoch", False, True, ["validation.csv"], [2]),
        ("Simple", "epoch", True, True, ["train.csv", "validation.csv"], [2, 2]),
        ("Simple", "epoch", False, False, [], []),
    ],
)
def test_csv_callback(
    backbone: str,
    save_interval: str,
    save_training: bool,
    save_validation: bool,
    expected_file_names: List,
    expected_csv_rows_nr: List,
):
    """
    2 epochs, 10 steps each
    5 steps validation
    """
    tmp_dir = "saved_files"

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)

    model = getSimpleModel()
    params = getParameters(backbone)
    dataset = getSimpleDataset()

    trainer = Trainer(model=model, cfg=params)

    csv_callback = CsvCallback(
        save_dir=tmp_dir, save_training=save_training, save_validation=save_validation, save_interval=save_interval
    )

    trainer.fit(
        train_dataset=dataset,
        batch_size=1,
        epochs=2,
        validation_dataset=dataset,
        validation_steps=5,
        validation_metric="acc",
        steps_per_epoch=10,
        callbacks=[csv_callback],
    )

    all_file_names = []
    row_number = []

    for r, d, files in os.walk(tmp_dir):
        all_file_names += files
        print("files ", files)

    print(
        "aaaaaa ",
    )

    # check if two file lists are the same
    assert collections.Counter(expected_file_names) == collections.Counter(all_file_names)

    for file in all_file_names:
        csv_file = pd.read_csv(os.path.join(tmp_dir, file))
        row_number.append(csv_file.shape[0])

    actual_row_nrs = dict(zip(all_file_names, row_number))
    desired_row_nrs = dict(zip(expected_file_names, expected_csv_rows_nr))

    assert actual_row_nrs == desired_row_nrs
    shutil.rmtree(tmp_dir)
