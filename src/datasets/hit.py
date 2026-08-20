"""
HIT Aerospace Engine Intershaft Bearing Dataset
HIT 航空发动机轴间轴承数据集

Dataset from Harbin Institute of Technology dual-rotor test platform.
数据来自哈尔滨工业大学双转子试验平台。

Data Structure:
- Sampling frequency: 25 kHz
- Sample length: 20480 points
- Channels: 6 (2 displacement + 4 acceleration)
- Format: .npy tensor N×6×20480

Subsets:
- data1: Normal bearing (28 conditions, 504 samples)
- data2: Normal bearing (25 conditions, 450 samples) 
- data3: Inner fault bearing A (28 conditions, 504 samples)
- data4: Inner fault bearing B (28 conditions, 504 samples)
- data5: Outer fault bearing (25 conditions, 450 samples)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.datamodule import TUDataModule
from src.datasets.base_dataset import dataset
from src.datasets.noise import NoisyEvaluationMixin
from src.datasets.transforms import build_transforms
from src.datasets.utils import create_train_val_split
from typing import Literal, List, Optional, Union

# Default signal size (can be overridden)
signal_size = 1024

# ID data: Normal + Inner fault
ID_FILES = ["data1.npy", "data3.npy"]
ID_LABELS = [0, 1]  # 0: Normal, 1: Inner fault

# Shift data: Different conditions/crack lengths
SHIFT_FILES = ["data2.npy", "data4.npy"]
SHIFT_LABELS = [0, 1]  # Same label mapping

# OOD data: Outer fault (unseen fault type)
OOD_FILES = ["data5.npy"]
OOD_LABEL = -1


def load_npy_data(
    filepath: Path, 
    label: int, 
    signal_size: int = 1024,
    channels: Optional[List[int]] = None
) -> tuple:
    """
    Load .npy file and slice samples.
    加载 .npy 文件并切片样本。
    
    Args:
        filepath: Path to .npy file
        label: Label for all samples in this file
        signal_size: Length of signal to extract (default 1024)
        channels: List of channel indices to use (0-5). 
                  None = all 6 channels.
                  Example: [0] for single channel, [0,1,2] for first 3 channels
    
    Returns:
        (data_list, label_list)
    """
    arr = np.load(filepath)  # N×6×20480
    data, labels = [], []
    
    # Default: use all channels
    if channels is None:
        channels = list(range(6))
    
    for sample in arr:  # sample: 6×20480
        # Select channels
        sample = sample[channels, :]  # len(channels)×20480
        
        # Truncate to signal_size
        if signal_size < sample.shape[1]:
            sample = sample[:, :signal_size]  # len(channels)×signal_size
        
        # Transpose to (L, C) to match transforms expectation
        # (L, C) -> Transpose -> (C, L) which is (channels, seq_len)
        sample = sample.T
        
        data.append(sample)
        labels.append(label)
    
    return data, labels


def build_df_from_files(
    root: Path, 
    file_list: List[str], 
    label_list: List[int],
    signal_size: int = 1024,
    channels: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Build DataFrame from multiple .npy files.
    从多个 .npy 文件构建 DataFrame。
    """
    all_data, all_labels = [], []
    
    for fname, lbl in zip(file_list, label_list):
        path = root / fname
        if not path.exists():
            print(f"Warning: {path} not found, skipping...")
            continue
        d, l = load_npy_data(path, lbl, signal_size, channels)
        all_data += d
        all_labels += l
    
    return pd.DataFrame({"data": all_data, "label": all_labels})


class HITDataModule(NoisyEvaluationMixin, TUDataModule):
    """
    HIT Aerospace Engine Intershaft Bearing DataModule.
    HIT 航空发动机轴间轴承数据模块。
    
    Args:
        root: Path to data directory (./data/hit)
        batch_size: Batch size for training
        channels: Channel indices to use. Options:
            - None: All 6 channels (default)
            - [0]: Single channel (channel 0)
            - [0, 1]: First 2 channels
            - [0, 1, 2, 3, 4, 5]: All channels explicitly
        signal_size: Length of signal per sample (default 1024)
        eval_ood: Whether to evaluate OOD detection
        eval_shift: Whether to evaluate domain shift
    
    Example:
        # Use all 6 channels
        dm = HITDataModule(root="./data/hit", batch_size=32)
        
        # Use single channel
        dm = HITDataModule(root="./data/hit", batch_size=32, channels=[0])
        
        # Use first 3 channels
        dm = HITDataModule(root="./data/hit", batch_size=32, channels=[0, 1, 2])
    """
    
    training_task = "classification"
    ood_datasets = ["hit_outer_fault_ood"]

    def __init__(
            self,
            root: str | Path,
            batch_size: int,
            channels: Optional[List[int]] = None,
            signal_size: int = 1024,
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
            normalize_type: str = "-1-1",
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

        # Channel selection
        self.channels = channels if channels is not None else list(range(6))
        self.signal_size = signal_size
        
        # Set class attributes based on channel selection
        self.num_channels = len(self.channels)
        self.input_shape = (self.num_channels, self.signal_size)
        self.num_classes = 2  # Normal, Inner fault
        
        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.eval_noise = eval_noise
        self.noise_configs = noise_configs or [
            (noise_type, severity)
            for noise_type in self.noise_params
            for severity in range(1, 6)
        ]
        self.split_seed = split_seed
        self.normalize_type = normalize_type

        self.train_transform = build_transforms("train", normalize=self.normalize_type)
        self.val_transform = build_transforms("val", normalize=self.normalize_type)
        self.test_transform = build_transforms("val", normalize=self.normalize_type)
        self.ood_transform = build_transforms("val", normalize=self.normalize_type)

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if getattr(self, "_noisy_mode", False):
            return
        if not self.val_split:
            raise ValueError("val_split must be positive to keep validation and test sets separate.")
        root = Path(self.root)

        # Build ID dataset
        id_df = build_df_from_files(
            root, ID_FILES, ID_LABELS, 
            self.signal_size, self.channels
        )
        id_df = id_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Train/test split (80/20)
        test_ratio = 0.2
        test_size = int(len(id_df) * test_ratio)
        self.test_df = id_df.iloc[:test_size].reset_index(drop=True)
        train_df = id_df.iloc[test_size:].reset_index(drop=True)

        if stage in ("fit", None):
            full = dataset(list_data=train_df, transform=self.train_transform)
            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full, self.val_split, self.test_transform, self.split_seed
                )
        if stage in ("test", None):
            full = dataset(list_data=train_df, transform=self.train_transform)
            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full, self.val_split, self.test_transform, self.split_seed
                )
            self.test = dataset(list_data=self.test_df, transform=self.test_transform)

        # OOD dataset: Outer fault
        if self.eval_ood:
            self.ood_df = build_df_from_files(
                root, OOD_FILES, [OOD_LABEL],
                self.signal_size, self.channels
            )
            self.ood = dataset(list_data=self.ood_df, transform=self.ood_transform)

        # Shift dataset: Different conditions
        if self.eval_shift:
            shift_df = build_df_from_files(
                root, SHIFT_FILES, SHIFT_LABELS,
                self.signal_size, self.channels
            )
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
