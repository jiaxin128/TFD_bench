import sys, warnings

warnings.filterwarnings('ignore')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import hashlib
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.datamodule import TUDataModule
from src.datasets.base_dataset import dataset
from src.datasets.noise import DEFAULT_NOISE_PARAMS as NOISE_PARAMS, NoisyEvaluationMixin
from src.datasets.transforms import build_transforms
from typing import Literal
from scipy.signal import decimate

signal_size = 1024

ID_FILES = ["N1_2500_0.csv", "PGC1_2500_0.csv", "PGS1_2500_0.csv",
            "PGTB1_2500_0.csv", "SGC1_2500_0.csv", "SGTB1_2500_0.csv"]
SHIFT_FILES = ["N1_3000_40.csv", "PGC1_3000_40.csv", "PGS1_3000_40.csv"]
OOD_FILES = [
    #   "ORS1_3000_40.csv",
    #   "IRS1_3000_40.csv",
    #   "BS1_3000_40.csv",
    #   "CC1_3000_40.csv",
    "ORS1_var.csv",
    "IRS1_var.csv",
    "CC1_var.csv",
]

# 这些文件采集时量纲为 g，需换算为 m/s² 与其余文件对齐（采集端问题，事后修正）。
# 有意与 OOD_FILES 解耦：单位换算是文件的固有属性，不应随 OOD 集合定义变化。
UNIT_G_FILES = {
    "ORS1_var.csv", "IRS1_var.csv", "CC1_var.csv",
    "ORS1_3000_40.csv", "IRS1_3000_40.csv",
    "BS1_3000_40.csv", "CC1_3000_40.csv",
}

ID_LABELS = list(range(len(ID_FILES)))
SHIFT_LABELS = ID_LABELS

# OOD 样本不参与分类，标签统一置 -1，避免被误当作合法类别索引。
OOD_LABEL = -1

# OOD 集合降采样：每文件保留的最大窗口数。None 表示不限制。
OOD_MAX_PER_FILE: int | None = 200

# 数据划分专用种子。与 seed_everything 的全局 RNG 完全解耦，
# 保证不同 model seed 拿到完全相同的 train / val / test。
SPLIT_SEED = 12345
TEST_RATIO = 0.2

# temporal 切分时，pool 与 test 之间丢弃的窗口数，切断时间上的相邻性。
SPLIT_GAP_WINDOWS = 1


CACHE_DIR = Path.home() / ".cache" / "tfd_mgb"
DECIMATE_Q = 4


# --------------------------------------------------------------------------- #
# 读取与切窗
# --------------------------------------------------------------------------- #

def load_signal_csv(filename: str, column: int = 1) -> np.ndarray:
    df = pd.read_csv(filename, usecols=[column])
    return df.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1, 1)


def slice_windows(arr: np.ndarray, label: int, win: int = signal_size):
    data, labels = [], []
    start, end = 0, win
    max_len = arr.shape[0]
    while end <= max_len:
        data.append(arr[start:end])
        labels.append(label)
        start += win
        end += win
    return data, labels


def subsample_uniform(items: list, n_max: int | None) -> list:
    """等间隔抽取至多 n_max 个元素，覆盖整段录制，完全确定性。"""
    if n_max is None or len(items) <= n_max:
        return items
    idx = np.linspace(0, len(items) - 1, n_max).round().astype(int)
    return [items[i] for i in idx]


def _cache_path(filename: str) -> Path:
    """缓存名包含源文件 mtime 和处理参数，源文件或参数变了自动失效。

    单位换算标记必须进 key：同一文件在换算与不换算两种情况下内容差 9.8 倍，
    否则会命中错误的缓存。
    """
    src = Path(filename)
    scale = "g98" if src.name in UNIT_G_FILES else "raw"
    key = f"{src.resolve()}|{src.stat().st_mtime_ns}|col1|q{DECIMATE_Q}|fir_zp|{scale}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{src.stem}_{h}.npy"


def load_decimated(filename: str) -> np.ndarray:
    """返回降采样后的 (N, 1) float32 数组，带磁盘缓存。"""
    cache = _cache_path(filename)
    if cache.exists():
        return np.load(cache)

    arr = load_signal_csv(filename, 1)
    if Path(filename).name in UNIT_G_FILES:
        arr = arr * 9.8            # g → m/s²，与其余文件量纲对齐
    arr = decimate(arr[:, 0], DECIMATE_Q, ftype="fir", zero_phase=True)
    arr = arr.astype(np.float32).reshape(-1, 1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.replace(cache)          # 原子替换，避免多进程写坏
    return arr


def data_load(filename: str, label: int):
    arr = load_decimated(filename)
    return slice_windows(arr, label, win=signal_size)


def build_df_from_files(root: str | Path, file_list, label_list,
                        max_per_file: int | None = None) -> pd.DataFrame:
    all_data, all_labels = [], []
    for fname, lbl in zip(file_list, label_list):
        path = Path(root) / fname
        d, l = data_load(str(path), lbl)
        if max_per_file is not None:
            d = subsample_uniform(d, max_per_file)
            l = l[:len(d)]
        all_data += d
        all_labels += l
    return pd.DataFrame({"data": all_data, "label": all_labels})


def build_split_by_time(root: str | Path, file_list, label_list,
                        test_ratio: float = TEST_RATIO,
                        gap_windows: int = SPLIT_GAP_WINDOWS):
    """按文件内时间顺序切分：每个文件前 (1-r) 段进 pool，后 r 段进 test。

    完全确定性，不使用随机数。切分点前丢弃 gap_windows 个窗作为缓冲，
    确保 pool 与 test 的窗口在时间上不相邻，避免同段录制导致的泄漏。
    """
    root = Path(root)
    pool_d, pool_l, test_d, test_l = [], [], [], []

    for fname, lbl in zip(file_list, label_list):
        d, l = data_load(str(root / fname), lbl)
        cut = int(len(d) * (1 - test_ratio))
        keep = max(cut - gap_windows, 0)
        pool_d += d[:keep]
        pool_l += l[:keep]
        test_d += d[cut:]
        test_l += l[cut:]

    return (pd.DataFrame({"data": pool_d, "label": pool_l}),
            pd.DataFrame({"data": test_d, "label": test_l}))


def stratified_split(df: pd.DataFrame, val_ratio: float, seed: int):
    """按类别分层切分，使用独立 RNG，不受全局随机状态影响。"""
    rng = np.random.default_rng(seed)
    val_positions = []
    labels = df["label"].to_numpy()
    for lbl in np.unique(labels):
        pos = np.flatnonzero(labels == lbl)
        rng.shuffle(pos)
        n_val = int(round(len(pos) * val_ratio))
        val_positions.append(pos[:n_val])
    val_pos = np.sort(np.concatenate(val_positions))
    mask = np.zeros(len(df), dtype=bool)
    mask[val_pos] = True
    val_df = df.iloc[mask].reset_index(drop=True)
    train_df = df.iloc[~mask].reset_index(drop=True)
    return train_df, val_df


def temporal_val_split(df: pd.DataFrame, val_ratio: float):
    """按类别在时间顺序上切 train/val：每类后 val_ratio 段作为 val。

    要求 df 内同类样本仍保持时间顺序（build_split_by_time 的输出满足）。
    """
    labels = df["label"].to_numpy()
    train_pos, val_pos = [], []
    for lbl in np.unique(labels):
        pos = np.flatnonzero(labels == lbl)      # 已按时间升序
        cut = int(round(len(pos) * (1 - val_ratio)))
        train_pos.append(pos[:cut])
        val_pos.append(pos[cut:])
    train_pos = np.sort(np.concatenate(train_pos))
    val_pos = np.sort(np.concatenate(val_pos))
    return (df.iloc[train_pos].reset_index(drop=True),
            df.iloc[val_pos].reset_index(drop=True))


# --------------------------------------------------------------------------- #
# DataModule
# --------------------------------------------------------------------------- #

class MGBDataModule(NoisyEvaluationMixin, TUDataModule):
    num_classes = 6
    num_channels = 1
    input_shape = (1, signal_size)
    training_task = "classification"
    ood_datasets = ["mgb_ood"]
    noise_params = NOISE_PARAMS

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
            normlize_type: str = "mean-std",
            pin_memory: bool = True,
            persistent_workers: bool = True,
            eval_noise: bool = False,
            noise_configs: list[tuple[str, int]] | None = None,
            split_seed: int = SPLIT_SEED,
            ood_max_per_file: int | None = OOD_MAX_PER_FILE,
            split_mode: Literal["random", "temporal"] = "random",
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
        self.noise_configs = noise_configs or [(t, s) for t in NOISE_PARAMS for s in range(1, 6)]
        self.normlize_type = normlize_type
        self.split_seed = split_seed
        self.ood_max_per_file = ood_max_per_file
        self.split_mode = split_mode
        self._split_done = False

        self.train_transform = build_transforms("train", normalize=self.normlize_type)
        self.val_transform = build_transforms("val", normalize=self.normlize_type)
        self.test_transform = build_transforms("val", normalize=self.normlize_type)
        self.ood_transform = build_transforms("val", normalize=self.normlize_type)

    # ----------------------------------------------------------------- #

    def _build_splits(self) -> None:
        """只执行一次。fit 与 test 阶段共用同一份划分。"""
        if self._split_done:
            return

        root = Path(self.root)

        if not self.val_split:
            raise ValueError(
                "val_split 必须为正数。验证集不能等于测试集，"
                "否则模型选择与后处理校准都会泄漏测试数据。"
            )

        if self.split_mode == "temporal":
            # 时间切分：pool / test 按录制顺序分开，中间留缓冲窗。
            # train / val 同样按时间切，保证三者互不相邻。
            pool_df, self.test_df = build_split_by_time(
                root, ID_FILES, ID_LABELS, TEST_RATIO)
            self.train_df, self.val_df = temporal_val_split(pool_df, self.val_split)
        else:
            id_df = build_df_from_files(root, ID_FILES, ID_LABELS)
            # 测试集划分固定为 random_state=42，与 model seed 无关
            id_df = id_df.sample(frac=1, random_state=42).reset_index(drop=True)
            test_size = int(len(id_df) * TEST_RATIO)
            self.test_df = id_df.iloc[:test_size].reset_index(drop=True)
            pool_df = id_df.iloc[test_size:].reset_index(drop=True)
            self.train_df, self.val_df = stratified_split(
                pool_df, self.val_split, self.split_seed)

        if self.eval_ood:
            self.ood_df = build_df_from_files(
                root, OOD_FILES, [OOD_LABEL] * len(OOD_FILES),
                max_per_file=self.ood_max_per_file)
        if self.eval_shift:
            self.shift_df = build_df_from_files(root, SHIFT_FILES, SHIFT_LABELS)

        self._split_done = True

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        # 噪声评估模式下跳过 setup，防止覆盖已替换的 test / ood
        if getattr(self, '_noisy_mode', False):
            return

        self._build_splits()

        self.train = dataset(list_data=self.train_df, transform=self.train_transform)
        self.val = dataset(list_data=self.val_df, transform=self.test_transform)

        if stage in ("test", None):
            self.test = dataset(list_data=self.test_df, transform=self.test_transform)

        if self.eval_ood:
            self.ood = dataset(list_data=self.ood_df, transform=self.ood_transform)

        if self.eval_shift:
            self.shift = dataset(list_data=self.shift_df, transform=self.test_transform)

    # ----------------------------------------------------------------- #

    def split_summary(self) -> str:
        """打印各划分的样本数与每类分布，用于确认划分不随 seed 变化。"""
        self._build_splits()
        lines = [f"split_mode={self.split_mode}  decimate_q={DECIMATE_Q}"]
        for name, df in [("train", self.train_df), ("val", self.val_df),
                         ("test", self.test_df)]:
            counts = df["label"].value_counts().sort_index().to_dict()
            lines.append(f"{name:6s} n={len(df):5d}  per-class={counts}")
        if self.eval_ood:
            lines.append(f"{'ood':6s} n={len(self.ood_df):5d}  "
                         f"max_per_file={self.ood_max_per_file}")
        if self.eval_shift:
            lines.append(f"{'shift':6s} n={len(self.shift_df):5d}")
        return "\n".join(lines)

    def test_dataloader(self) -> list[DataLoader]:
        dataloaders = [self._data_loader(self.get_test_set(), training=False, shuffle=False)]
        if self.eval_ood:
            dataloaders.append(
                self._data_loader(self.get_ood_set(), training=False, shuffle=False))
        if self.eval_shift:
            dataloaders.append(
                self._data_loader(self.get_shift_set(), training=False, shuffle=False))
        return dataloaders


if __name__ == "__main__":
    import torch

    for mode in ["random", "temporal"]:
        print(f"\n{'=' * 60}\n  split_mode = {mode}\n{'=' * 60}")
        fingerprints = []
        for s in [0, 1, 2]:
            torch.manual_seed(s)
            np.random.seed(s)
            dm = MGBDataModule(root="/mnt/d/Data/Machine/MGB", batch_size=64,
                               val_split=0.2, eval_ood=True, split_mode=mode)
            dm.setup("test")
            print(f"\n--- global seed {s} ---")
            print(dm.split_summary())
            fp = hashlib.md5(
                str(dm.val_df["label"].tolist()
                    + [float(a.sum()) for a in dm.val_df["data"]]).encode()
            ).hexdigest()[:12]
            fingerprints.append(fp)
            print("val fingerprint:", fp)

        print("\n划分是否稳定:",
              "是" if len(set(fingerprints)) == 1 else "否 —— 仍受全局随机状态影响")
