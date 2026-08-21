# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""
PU Bearing Dataset / 帕德博恩大学轴承数据集

Paderborn University Bearing Data Center Dataset.
64kHz sampling rate with multiple fault types and operating conditions.

Dataset structure:
    pu/
    └── PU-dataset-main/
        ├── N09_M07_F10/  (转速900rpm, 负载矩0.7Nm, 径向力10N)
        │   ├── N09_M07_F10_K001_1.mat
        │   ├── N09_M07_F10_KA04_1.mat
        │   └── ...
        ├── N15_M01_F10/  (转速1500rpm, 负载矩0.1Nm, 径向力10N)
        ├── N15_M07_F04/  (转速1500rpm, 负载矩0.7Nm, 径向力4N)
        └── N15_M07_F10/  (转速1500rpm, 负载矩0.7Nm, 径向力10N)

Reference: https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter

Classes / 类别:
    ID (13 classes): KA04, KA15, KA16, KA22, KA30, KB23, KB24, KB27, KI04, KI16, KI17, KI18, KI21
    OOD: K001-K006 (正常健康状态 - 作为OOD)
    Shift: 不同工况下的相同类别 (例如从N15_M07_F10到N09_M07_F10)
    
Fault Types / 故障类型:
    - K001-K006: 健康轴承 (6 types)
    - KA系列: 外圈故障 (Outer Race - Außenring)
    - KI系列: 内圈故障 (Inner Race - Innenring)
    - KB系列: 滚动体故障 (Rolling Element - Rollkörper)
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
from typing import Literal, List, Optional

signal_size = 1024

# =============================================================================
# Working Conditions / 工况定义
# =============================================================================

# ID工况: N15_M07_F10 (转速1500rpm, 负载矩0.7Nm, 径向力10N) - 最常用工况
ID_CONDITION = "N15_M07_F10"

# Shift工况: 不同的操作条件
SHIFT_CONDITIONS = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]

# 所有工况
ALL_CONDITIONS = [ID_CONDITION] + SHIFT_CONDITIONS

# =============================================================================
# Fault Type Mapping / 故障类型映射
# =============================================================================

# ID类别: 外圈(KA) + 内圈(KI) + 滚动体(KB) 故障类别
ID_CLASSES = [
    "K001", "K002", "K003", "K004", "K005", "K006",  # 健康轴承 (6)
    "KA04", "KA15", "KA16", "KA22", "KA30",  # 外圈故障 (5)
    "KB23", "KB24", "KB27",                    # 滚动体故障 (3)
]
ID_LABELS = {cls: i for i, cls in enumerate(ID_CLASSES)}
NUM_ID_CLASSES = len(ID_CLASSES)

# OOD类别: 健康轴承 (训练时未见)
OOD_CLASSES = ["KI04", "KI16", "KI17", "KI18", "KI21"]  # 内圈故障 (5)
OOD_LABEL = -1
OOD_LABELS = {cls: OOD_LABEL for cls in OOD_CLASSES}

# 所有轴承类别
ALL_BEARING_CLASSES = ID_CLASSES + OOD_CLASSES


def load_mat_file(filepath: str) -> np.ndarray:
    """
    Load vibration signal from PU .mat file.
    从PU .mat文件加载振动信号。
    
    PU dataset MAT file structure:
        data[main_key][0,0]['Y'][0, channel_idx]['Data']
    
    The vibration data is in the channel named 'vibration_1'.
    """
    mat_data = sio.loadmat(filepath)
    
    # Get the main key (e.g., 'N15_M07_F10_KA04_1')
    main_key = None
    for key in mat_data.keys():
        if not key.startswith('_'):
            main_key = key
            break
    
    if main_key is None:
        raise ValueError(f"Could not find main key in {filepath}")
    
    try:
        # Navigate the nested structure: data[main_key][0,0]['Y']
        inner = mat_data[main_key][0, 0]
        Y = inner['Y']  # Shape: (1, num_channels)
        
        # Search for the vibration channel
        for i in range(Y.shape[1]):
            channel = Y[0, i]
            name = channel['Name'].flatten()[0] if channel['Name'].size > 0 else ''
            
            if 'vibration' in name.lower():
                vibration_data = channel['Data'].flatten()
                return vibration_data
        
        raise ValueError(f"Could not find vibration channel in {filepath}")
        
    except Exception as e:
        # Fallback: try the old method for backwards compatibility
        for key in mat_data.keys():
            if key.startswith('_'):
                continue
            arr = mat_data[key]
            if isinstance(arr, np.ndarray):
                if arr.dtype == np.object_:
                    try:
                        inner = arr[0, 0]
                        for i in range(len(inner)):
                            if isinstance(inner[i], np.ndarray) and inner[i].dtype in [np.float64, np.float32]:
                                if inner[i].size > 1000:
                                    return inner[i].flatten()
                    except (IndexError, TypeError):
                        pass
                elif arr.size > 1000:
                    return arr.flatten()
        
        raise ValueError(f"Could not find vibration data in {filepath}: {e}")


def slice_windows(arr: np.ndarray, label: int, win: int = signal_size):
    """Slice signal into fixed-length windows / 将信号切分为固定长度窗口"""
    data, labels = [], []
    start, end = 0, win
    max_len = arr.shape[0]

    while end <= max_len:
        segment = arr[start:end].reshape(-1, 1)  # Shape: (win, 1)
        data.append(segment)
        labels.append(label)
        start += win
        end += win

    return data, labels


def data_load(filepath: str, label: int):
    """Load and slice data from a .mat file / 从.mat文件加载并切分数据"""
    arr = load_mat_file(filepath)
    return slice_windows(arr, label, win=signal_size)


def get_file_path(root: Path, condition: str, bearing_class: str) -> Path:
    """
    Get the file path for a specific condition and bearing class.
    获取特定工况和轴承类别的文件路径。
    
    File naming convention: {condition}_{bearing_class}_1.mat
    """
    return root / "PU-dataset-main" / condition / f"{condition}_{bearing_class}_1.mat"


def build_df_from_files(
    root: Path, 
    condition: str, 
    classes: List[str], 
    label_dict: dict
) -> pd.DataFrame:
    """
    Build DataFrame from files for a specific condition.
    从特定工况的文件构建DataFrame。
    """
    all_data, all_labels, all_sources = [], [], []
    
    for bearing_class in classes:
        filepath = get_file_path(root, condition, bearing_class)
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        label = label_dict.get(bearing_class, 0)
        
        try:
            d, l = data_load(str(filepath), label)
            all_data += d
            all_labels += l
            all_sources += [bearing_class] * len(d)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return pd.DataFrame({
        "data": all_data,
        "label": all_labels,
        "source": all_sources,
    })


def build_df_from_multiple_conditions(
    root: Path,
    conditions: List[str],
    classes: List[str],
    label_dict: dict
) -> pd.DataFrame:
    """
    Build DataFrame from files across multiple conditions.
    从多个工况的文件构建DataFrame。
    """
    dfs = []
    for condition in conditions:
        df = build_df_from_files(root, condition, classes, label_dict)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


class PUDataModule(NoisyEvaluationMixin, TUDataModule):
    """
    PU Bearing Dataset DataModule.
    帕德博恩大学轴承数据集 DataModule。
    
    ID: 13 fault classes (KA04, KA15, KA16, KA22, KA30, KB23, KB24, KB27, 
                          KI04, KI16, KI17, KI18, KI21) under N15_M07_F10 condition
    Shift: Same 13 classes under different operating conditions
    OOD: K001-K006 (Normal/Healthy bearings - unseen in training)
    """
    
    num_classes = NUM_ID_CLASSES  # 13 classes
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["pu_healthy_ood"]

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
        
        # Build ID data (N15_M07_F10 condition)
        id_df = build_df_from_files(root, ID_CONDITION, ID_CLASSES, ID_LABELS)
        id_df = id_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Train/test split
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

        # OOD data: Healthy bearings K001-K006 (never seen in training)
        if self.eval_ood:
            self.ood_df = build_df_from_files(root, ID_CONDITION, OOD_CLASSES, OOD_LABELS)
            self.ood = dataset(list_data=self.ood_df, transform=self.ood_transform)

        # Shift data: Same classes under different operating conditions
        if self.eval_shift:
            shift_df = build_df_from_multiple_conditions(
                root, SHIFT_CONDITIONS, ID_CLASSES, ID_LABELS
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


# =============================================================================
# Utility functions / 工具函数
# =============================================================================

def print_dataset_info(root: str | Path):
    """Print dataset information / 打印数据集信息"""
    root = Path(root)
    
    print("=" * 60)
    print("PU Bearing Dataset Info / PU轴承数据集信息")
    print("=" * 60)
    
    # Check available files
    for condition in ALL_CONDITIONS:
        print(f"\nCondition / 工况: {condition}")
        print("-" * 40)
        
        available_id = []
        missing_id = []
        for cls in ID_CLASSES:
            filepath = get_file_path(root, condition, cls)
            if filepath.exists():
                available_id.append(cls)
            else:
                missing_id.append(cls)
        
        available_ood = []
        missing_ood = []
        for cls in OOD_CLASSES:
            filepath = get_file_path(root, condition, cls)
            if filepath.exists():
                available_ood.append(cls)
            else:
                missing_ood.append(cls)
        
        print(f"  ID classes available: {len(available_id)}/{len(ID_CLASSES)}")
        print(f"  OOD classes available: {len(available_ood)}/{len(OOD_CLASSES)}")
        
        if missing_id:
            print(f"  Missing ID: {missing_id}")
        if missing_ood:
            print(f"  Missing OOD: {missing_ood}")
    
    print("\n" + "=" * 60)
    print(f"ID Classes ({len(ID_CLASSES)} total):")
    print(f"  Outer Race (KA): {[c for c in ID_CLASSES if c.startswith('KA')]}")
    print(f"  Rolling Element (KB): {[c for c in ID_CLASSES if c.startswith('KB')]}")
    print(f"  Inner Race (KI): {[c for c in ID_CLASSES if c.startswith('KI')]}")
    print(f"\nOOD Classes ({len(OOD_CLASSES)} total):")
    print(f"  Healthy (K): {OOD_CLASSES}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Default path or command line argument
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = r".\data\pu"
    
    print_dataset_info(data_root)
    
    # Test loading
    print("\nTesting DataModule...")
    dm = PUDataModule(root=data_root, batch_size=32, eval_ood=True, eval_shift=True)
    dm.setup()
    
    print(f"Train samples: {len(dm.train)}")
    print(f"Val samples: {len(dm.val)}")
    print(f"Test samples: {len(dm.test)}")
    if dm.eval_ood:
        print(f"OOD samples: {len(dm.ood)}")
    if dm.eval_shift:
        print(f"Shift samples: {len(dm.shift)}")
