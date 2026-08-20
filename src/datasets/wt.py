"""
WT (Wind Turbine) Gearbox Dataset Module.

Dataset structure:
- 5 fault categories: broken, healthy, missing_tooth, root_crack, wear
- Each category has files with naming: {FaultPrefix}{Assembly}_{Speed}.MAT
    - Assembly: 1 or 2 (two different assemblies/拆装)
    - Speed: 20, 25, 30, 35, 40, 45, 50, 55 Hz
- Each MAT file contains 'Data' with shape (N, 4) - 4 channels
- Sample rate: 48000 Hz

Data Split Strategy:
================================================================================
- ID (In-Distribution): 3 fault types (broken, healthy, missing_tooth)
                        Assembly 1, speeds 30-40 Hz
- OOD (Out-of-Distribution): 2 fault types (root_crack, wear)
                             Unseen fault categories during training
- Shift: Same 3 ID fault types, but with different conditions:
         Assembly 2 (different installation) + different speeds (20, 25, 50, 55 Hz)

Examples for customization:
================================================================================
If you want to change the shift conditions, modify the following variables:

1. Assembly shift only (不同拆装):
   SHIFT_FILES = generate_file_list(ID_FAULT_CATEGORIES, assembly=2, speeds=[30, 35, 40])

2. Speed shift only (不同转速):
   SHIFT_FILES = generate_file_list(ID_FAULT_CATEGORIES, assembly=1, speeds=[20, 25, 50, 55])

3. Both assembly and speed shift (当前默认配置):
   SHIFT_FILES = (generate_file_list(ID_FAULT_CATEGORIES, assembly=2, speeds=[30, 35, 40]) +
                  generate_file_list(ID_FAULT_CATEGORIES, assembly=1, speeds=[20, 25, 50, 55]))
"""
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.io as sio
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.datamodule import TUDataModule
from src.datasets.base_dataset import dataset
from src.datasets.noise import NoisyEvaluationMixin
from src.datasets.transforms import build_transforms
from src.datasets.utils import create_train_val_split
from typing import Literal

signal_size = 1024

# ============================================================================
# Fault category definitions
# ============================================================================
# ID fault categories (used for training and testing)
ID_FAULT_CATEGORIES = {
    "broken": "B",
    "healthy": "N",
    "missing_tooth": "M",
}

# OOD fault categories (unseen during training)
OOD_FAULT_CATEGORIES = {
    "root_crack": "R",
    "wear": "W",
}

# ============================================================================
# File list generation
# ============================================================================
def generate_file_list(fault_categories: dict, assembly: int, speeds: list):
    """Generate file tuples (folder, filename, label) for given parameters."""
    files = []
    for label, (folder, prefix) in enumerate(fault_categories.items()):
        for speed in speeds:
            filename = f"{prefix}{assembly}_{speed}.MAT"
            files.append((folder, filename, label))
    return files


# ID: 3 fault types, Assembly 1, speeds 30-40 Hz
ID_FILES = generate_file_list(ID_FAULT_CATEGORIES, assembly=1, speeds=[30, 35, 40])

# Shift: Same 3 ID fault types, Assembly 2 + different speeds
SHIFT_FILES = (
    generate_file_list(ID_FAULT_CATEGORIES, assembly=2, speeds=[30, 35, 40]) +
    generate_file_list(ID_FAULT_CATEGORIES, assembly=1, speeds=[20, 25, 50, 55])
)

# OOD: Unseen fault types (root_crack, wear)
OOD_FILES = (
    generate_file_list(OOD_FAULT_CATEGORIES, assembly=1, speeds=[30, 35, 40]) +
    generate_file_list(OOD_FAULT_CATEGORIES, assembly=2, speeds=[30, 35, 40])
)

ID_LABELS = list(range(len(ID_FAULT_CATEGORIES)))  # 0~2 (3 classes)
SHIFT_LABELS = ID_LABELS  # Shift has the same 3 classes as ID
OOD_LABEL = -1

# Channel to use (0-3, using channel 0 by default)
DEFAULT_CHANNEL = 0


# ============================================================================
# Data loading functions
# ============================================================================
def load_signal_mat(filepath: str, channel: int = DEFAULT_CHANNEL) -> np.ndarray:
    """Load signal data from a MAT file."""
    mat = sio.loadmat(filepath)
    data = mat['Data']  # Shape: (N, 4)
    signal = data[:, channel]
    return signal.reshape(-1, 1)


def slice_windows(arr: np.ndarray, label: int, win: int = signal_size):
    """Slice the signal into non-overlapping windows."""
    data, labels = [], []
    start, end = 0, win
    max_len = arr.shape[0]

    while end <= max_len:
        data.append(arr[start:end])
        labels.append(label)
        start += win
        end += win

    return data, labels


def data_load(filepath: str, label: int, channel: int = DEFAULT_CHANNEL):
    """Load data from a MAT file and slice into windows."""
    arr = load_signal_mat(filepath, channel)
    return slice_windows(arr, label, win=signal_size)


def build_df_from_files(root: str | Path, file_list: list):
    """Build a DataFrame from a list of file tuples."""
    root = Path(root)
    all_data, all_labels = [], []

    for folder, filename, label in file_list:
        filepath = root / folder / filename
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        d, l = data_load(str(filepath), label)
        all_data += d
        all_labels += l

    return pd.DataFrame({"data": all_data, "label": all_labels})


# ============================================================================
# DataModule class
# ============================================================================
class WTDataModule(NoisyEvaluationMixin, TUDataModule):
    num_classes = 3  # ID fault types only
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["wt_ood"]

    def __init__(
            self,
            root: str | Path,
            batch_size: int,
            eval_batch_size: int | None = None,
            eval_ood: bool = False,
            eval_shift: bool = False,
            num_tta: int = 1,
            val_split: float | None = None,
            postprocess_set: Literal["val", "test"] = "val",
            num_workers: int = 1,
            train_transform: nn.Module | None = None,
            test_transform: nn.Module | None = None,
            ood_transform: nn.Module | None = None,
            normlize_type: str = "-1-1",
            pin_memory: bool = True,
            persistent_workers: bool = True,
            eval_noise: bool = False,
            noise_configs: list[tuple[str, int]] | None = None,
            split_seed: int = 12345,
    ) -> None:

        super().__init__(
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_tta=num_tta,
            postprocess_set=postprocess_set,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.eval_noise = eval_noise
        self.noise_configs = noise_configs or [
            (noise_type, severity)
            for noise_type in self.noise_params
            for severity in range(1, 6)
        ]
        self.split_seed = split_seed
        self.normlize_type = normlize_type

        self.train_transform = build_transforms("train", normalize=self.normlize_type)
        self.val_transform = build_transforms("val", normalize=self.normlize_type)
        self.test_transform = build_transforms("val", normalize=self.normlize_type)
        self.ood_transform = build_transforms("val", normalize=self.normlize_type)

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if getattr(self, "_noisy_mode", False):
            return
        if not self.val_split:
            raise ValueError("val_split must be positive to keep validation and test sets separate.")
        root = Path(self.root)

        id_df = build_df_from_files(root, ID_FILES)
        id_df = id_df.sample(frac=1, random_state=42).reset_index(drop=True)
        test_ratio = 0.2
        test_size = int(len(id_df) * test_ratio)
        self.test_df = id_df.iloc[:test_size].reset_index(drop=True)
        train_df = id_df.iloc[test_size:].reset_index(drop=True)

        if stage in ("fit", None):
            full = dataset(list_data=train_df, transform=self.train_transform)
            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full, self.val_split,
                    self.test_transform, self.split_seed)
        if stage in ("test", None):
            full = dataset(list_data=train_df, transform=self.train_transform)
            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full, self.val_split,
                    self.test_transform, self.split_seed)
            self.test = dataset(list_data=self.test_df, transform=self.test_transform)

        if self.eval_ood:
            self.ood_df = build_df_from_files(root, OOD_FILES)
            self.ood_df["label"] = OOD_LABEL
            self.ood = dataset(list_data=self.ood_df, transform=self.ood_transform)

        if self.eval_shift:
            shift_df = build_df_from_files(root, SHIFT_FILES)
            self.shift = dataset(list_data=shift_df, transform=self.test_transform)

    def test_dataloader(self) -> list[DataLoader]:
        dataloaders = [
            self._data_loader(self.get_test_set(), training=False, shuffle=False)
        ]
        if self.eval_ood:
            dataloaders.append(
                self._data_loader(self.get_ood_set(), training=False, shuffle=False)
            )
        if self.eval_shift:
            dataloaders.append(
                self._data_loader(self.get_shift_set(), training=False, shuffle=False)
            )
        return dataloaders
