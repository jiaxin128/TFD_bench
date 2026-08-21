# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
from torch.utils.data import Dataset
from src.datasets.transforms import *

import torch
from torch.utils.data import Dataset


class dataset(Dataset):

    def __init__(self, list_data, transform=None, test=False):
        super().__init__()

        # 支持 pandas DataFrame
        if hasattr(list_data, "loc"):
            self.data_list = list_data["data"].tolist()
            self.label_list = list_data["label"].tolist()

        # 支持 dict
        elif isinstance(list_data, dict):
            self.data_list = list_data["data"]
            self.label_list = list_data["label"]

        else:
            raise TypeError(
                f"list_data must be a DataFrame or dict. Got {type(list_data)}"
            )

        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        x = self.data_list[idx]
        y = self.label_list[idx]

        # 数据增强
        if self.transform is not None:
            x = self.transform(x)

        return x, y
