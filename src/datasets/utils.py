# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
import copy
from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset, random_split


def create_train_val_split(
    dataset: Dataset,
    val_split_rate: float,
    val_transforms: Callable | None = None,
    seed: int = 12345,
) -> tuple[Dataset, Dataset]:
    """Split a dataset for training and validation.

    Args:
        dataset (Dataset): The dataset to be split.
        val_split_rate (float): The amount of the original dataset to use as validation split.
        val_transforms (Callable | None, optional): The transformations to apply on the validation set.
            Defaults to ``None``.
        seed: Fixed split seed, independent from the model-training seed.

    Returns:
        tuple[Dataset, Dataset]: The training and the validation splits.
    """
    generator = torch.Generator().manual_seed(seed)
    train, val = random_split(
        dataset,
        [1 - val_split_rate, val_split_rate],
        generator=generator,
    )
    val = copy.deepcopy(val)  # Ensure train.dataset.transform is not modified next line
    val.dataset.transform = val_transforms
    return train, val


class TTADataset(Dataset):
    def __init__(self, dataset: Dataset, num_augmentations: int) -> None:
        """Create a version of the dataset that returns the same sample multiple times.

        This is useful for test-time augmentation (TTA).

        Args:
            dataset (Dataset): The dataset to be adapted for TTA.
            num_augmentations (int): The number of augmentations to apply.
        """
        super().__init__()
        self.dataset = dataset
        self.num_augmentations = num_augmentations

    def __len__(self) -> int:
        """Get the virtual length of the dataset."""
        return len(self.dataset) * self.num_augmentations

    def __getitem__(self, index) -> Any:
        """Get the item corresponding to idx // :attr:`self.num_augmentations`."""
        return self.dataset[index // self.num_augmentations]
