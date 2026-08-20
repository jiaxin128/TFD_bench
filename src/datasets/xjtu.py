"""
XJTU Gearbox Dataset / 西安交通大学齿轮箱数据集

XJTU Gearbox and Bearing Fault Diagnosis Dataset.
Includes bearing faults and planetary gear faults.

Dataset structure:
    xjtu/
    ├── 1ndBearing_ball/
    │   ├── Data_Chan1.txt
    │   └── Data_Chan2.txt
    ├── 1ndBearing_inner/
    ├── 1ndBearing_mix(inner+outer+ball)/
    ├── 1ndBearing_outer/
    ├── 2ndPlanetary_brokentooth/
    ├── 2ndPlanetary_missingtooth/
    ├── 2ndPlanetary_normalstate/
    ├── 2ndPlanetary_rootcracks/
    └── 2ndPlanetary_toothwear/

Classes / 类别:
    ID (9 classes): All fault types using Channel 1
    Shift: Same classes using Channel 2 (different sensor position)
    OOD: Can be configured (e.g., specific fault types unseen in training)

Fault Types / 故障类型:
    - 1ndBearing_*: 轴承故障 (ball, inner, outer, mix)
    - 2ndPlanetary_*: 行星齿轮故障 (brokentooth, missingtooth, normalstate, rootcracks, toothwear)
"""

from itertools import islice
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
from typing import Literal, List, Optional

signal_size = 1024

# =============================================================================
# Class Definitions / 类别定义
# =============================================================================

# All fault classes
ALL_CLASSES = [
    "1ndBearing_ball",
    "1ndBearing_inner",
    "1ndBearing_mix(inner+outer+ball)",
    "1ndBearing_outer",
    "2ndPlanetary_brokentooth",
    "2ndPlanetary_missingtooth",
    "2ndPlanetary_normalstate",
    "2ndPlanetary_rootcracks",
    "2ndPlanetary_toothwear",
]

# ID类别: 全部9类
ID_CLASSES = ALL_CLASSES.copy()
ID_LABELS = {cls: i for i, cls in enumerate(ID_CLASSES)}
NUM_ID_CLASSES = len(ID_CLASSES)

# OOD类别: 可以根据需要配置，这里使用mix类型作为OOD
# 因为mix类型是复合故障，更难识别
OOD_CLASSES = ["1ndBearing_mix(inner+outer+ball)"]
OOD_LABEL = -1

# ID类别(排除OOD)
ID_CLASSES_NO_OOD = [c for c in ID_CLASSES if c not in OOD_CLASSES]
ID_LABELS_NO_OOD = {cls: i for i, cls in enumerate(ID_CLASSES_NO_OOD)}

# Channel definitions
CHANNEL_ID = "Data_Chan1.txt"      # ID数据使用通道1
CHANNEL_SHIFT = "Data_Chan2.txt"   # Shift数据使用通道2

# Header lines to skip in txt files
HEADER_LINES = 14


def load_signal_txt(filepath: str) -> np.ndarray:
    """
    Load vibration signal from XJTU txt file.
    从XJTU txt文件加载振动信号。
    
    The file has a header section followed by numerical data.
    """
    data = []
    with open(filepath, "r", errors="ignore") as f:
        for line in islice(f, HEADER_LINES, None):
            try:
                val = float(line.strip())
                data.append(val)
            except ValueError:
                continue
    
    return np.array(data).reshape(-1, 1)


def slice_windows(arr: np.ndarray, label: int, win: int = signal_size):
    """Slice signal into fixed-length windows / 将信号切分为固定长度窗口"""
    data, labels = [], []
    start, end = 0, win
    max_len = arr.shape[0]

    while end <= max_len:
        data.append(arr[start:end])
        labels.append(label)
        start += win
        end += win

    return data, labels


def data_load(filepath: str, label: int):
    """Load and slice data from a txt file / 从txt文件加载并切分数据"""
    arr = load_signal_txt(filepath)
    return slice_windows(arr, label, win=signal_size)


def build_df_from_classes(
    root: Path, 
    classes: List[str], 
    label_dict: dict,
    channel: str = CHANNEL_ID
) -> pd.DataFrame:
    """
    Build DataFrame from class folders.
    从类别文件夹构建DataFrame。
    """
    all_data, all_labels = [], []
    
    for cls in classes:
        filepath = root / cls / channel
        
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue
        
        label = label_dict.get(cls, 0)
        
        try:
            d, l = data_load(str(filepath), label)
            all_data += d
            all_labels += l
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return pd.DataFrame({"data": all_data, "label": all_labels})


class XJTUDataModule(NoisyEvaluationMixin, TUDataModule):
    """
    XJTU Gearbox Dataset DataModule.
    西安交通大学齿轮箱数据集 DataModule。
    
    ID: 8 fault classes (excluding mix) using Channel 1
    Shift: Same 8 classes using Channel 2 (different sensor)
    OOD: 1ndBearing_mix (compound fault - unseen in training)
    """
    
    num_classes = len(ID_CLASSES_NO_OOD)  # 8 classes (excluding OOD)
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["xjtu_mix_fault_ood"]

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
        
        # Build ID data (Channel 1, excluding OOD classes)
        id_df = build_df_from_classes(root, ID_CLASSES_NO_OOD, ID_LABELS_NO_OOD, CHANNEL_ID)
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

        # OOD data: Mix fault (compound fault - never seen in training)
        if self.eval_ood:
            ood_labels = {cls: OOD_LABEL for cls in OOD_CLASSES}
            self.ood_df = build_df_from_classes(root, OOD_CLASSES, ood_labels, CHANNEL_ID)
            self.ood = dataset(list_data=self.ood_df, transform=self.ood_transform)

        # Shift data: Same classes using Channel 2 (different sensor)
        if self.eval_shift:
            shift_df = build_df_from_classes(root, ID_CLASSES_NO_OOD, ID_LABELS_NO_OOD, CHANNEL_SHIFT)
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
    print("XJTU Gearbox Dataset Info / XJTU齿轮箱数据集信息")
    print("=" * 60)
    
    print(f"\nID Classes ({len(ID_CLASSES_NO_OOD)} total):")
    for cls in ID_CLASSES_NO_OOD:
        chan1 = root / cls / CHANNEL_ID
        chan2 = root / cls / CHANNEL_SHIFT
        status1 = "✓" if chan1.exists() else "✗"
        status2 = "✓" if chan2.exists() else "✗"
        print(f"  {cls}: Chan1={status1}, Chan2={status2}")
    
    print(f"\nOOD Classes ({len(OOD_CLASSES)} total):")
    for cls in OOD_CLASSES:
        chan1 = root / cls / CHANNEL_ID
        status = "✓" if chan1.exists() else "✗"
        print(f"  {cls}: {status}")
    
    print("\n" + "-" * 40)
    print("Bearing Faults (轴承故障):")
    bearing = [c for c in ALL_CLASSES if c.startswith("1nd")]
    for c in bearing:
        print(f"  - {c}")
    
    print("\nPlanetary Gear Faults (行星齿轮故障):")
    planetary = [c for c in ALL_CLASSES if c.startswith("2nd")]
    for c in planetary:
        print(f"  - {c}")
    
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Default path or command line argument
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = r".\data\xjtu"
    
    print_dataset_info(data_root)
    
    # Test loading
    print("\nTesting DataModule...")
    dm = XJTUDataModule(root=data_root, batch_size=32, eval_ood=True, eval_shift=True)
    dm.setup()
    
    print(f"Train samples: {len(dm.train)}")
    print(f"Val samples: {len(dm.val)}")
    print(f"Test samples: {len(dm.test)}")
    if dm.eval_ood:
        print(f"OOD samples: {len(dm.ood)}")
    if dm.eval_shift:
        print(f"Shift samples: {len(dm.shift)}")
