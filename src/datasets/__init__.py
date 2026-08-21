# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""One-dimensional time-domain datasets for fault diagnosis."""

from src.datasets.seu import SEUDataModule
from src.datasets.wt import WTDataModule
from src.datasets.pu import PUDataModule
from src.datasets.xjtu import XJTUDataModule
from src.datasets.hit import HITDataModule
from src.datasets.cwru import CWRUDataModule
from src.datasets.thu import THUDataModule
from src.datasets.mgb import MGBDataModule
from src.datasets.base_dataset import dataset

__all__ = [
    "SEUDataModule",
    "WTDataModule",
    "PUDataModule",
    "XJTUDataModule",
    "HITDataModule",
    "CWRUDataModule",
    "THUDataModule",
    "MGBDataModule",
    "dataset",
]
