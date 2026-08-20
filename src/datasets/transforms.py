import numpy as np
import random
from scipy.signal import resample

from src.datasets.noise import AddGaussian


# ---------------------------
# Utility
# ---------------------------
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------
# Compose: Transform Pipeline
# ---------------------------
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seq):
        for t in self.transforms:
            seq = t(seq)
        return seq


# ---------------------------
# Basic Preprocessing
# ---------------------------

class Transpose(object):
    def __call__(self, x):
        # x shape: (L, C) → (C, L)
        return x.transpose(1, 0)


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class Normalize(object):
    def __init__(self, mode="0-1"):   # "0-1", "-1-1", "mean-std"
        self.mode = mode

    def __call__(self, seq):
        if self.mode == "0-1":
            seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-8)
        elif self.mode == "-1-1":
            seq = 2 * (seq - seq.min()) / (seq.max() - seq.min() + 1e-8) - 1
        elif self.mode == "mean-std":
            seq = (seq - seq.mean()) / (seq.std() + 1e-8)
        else:
            raise NameError("Unsupported normalization!")

        return seq


# ---------------------------
# Corruption and domain shift
# ---------------------------


class RandomMask(object):
    """Random zero-out segments (stress test for robustness)."""
    def __init__(self, mask_ratio=0.05):
        self.mask_ratio = mask_ratio

    def __call__(self, seq):
        L = seq.shape[1]
        mask_len = int(L * self.mask_ratio)
        start = np.random.randint(0, L - mask_len)
        seq[:, start:start+mask_len] = 0
        return seq


class AmplitudeShift(object):
    """Domain shift: amplitude scaling."""
    def __call__(self, seq):
        factor = np.random.uniform(0.8, 1.2)
        return seq * factor


# ---------------------------
# Augmentations
# ---------------------------

class Scale(object):
    def __call__(self, seq):
        factor = np.random.normal(1, 0.01, size=(seq.shape[0], 1))
        return seq * factor


class RandomScale(object):
    def __call__(self, seq):
        if np.random.rand() > 0.5:
            return seq
        factor = np.random.normal(1, 0.01, size=(seq.shape[0], 1))
        return seq * factor


class RandomStretch(object):
    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, seq):
        if np.random.rand() > 0.5:
            return seq
        seq_aug = np.zeros(seq.shape)
        L = seq.shape[1]
        length = int(L * (1 + (random.random() - 0.5) * self.sigma))
        for i in range(seq.shape[0]):
            y = resample(seq[i, :], length)
            if length < L:
                seq_aug[i, :length] = y
            else:
                seq_aug[i, :] = y[:L]
        return seq_aug


class RandomCrop(object):
    def __init__(self, crop_len=20):
        self.crop_len = crop_len

    def __call__(self, seq):
        if np.random.rand() > 0.5:
            return seq
        L = seq.shape[1]
        start = np.random.randint(0, L - self.crop_len)
        seq[:, start:start+self.crop_len] = 0
        return seq


# ---------------------------
# Benchmark-level Transform Builders
# ---------------------------

def build_transforms(mode="train", normalize="0-1"):
    """Benchmark-style builder."""
    if mode == "train":
        return Compose([
            Transpose(),
            Normalize(normalize),
            Retype()
        ])

    elif mode == "val":
        return Compose([
            Transpose(),
            Normalize(normalize),
            Retype()
        ])

    elif mode == "noise":
        return Compose([
            Transpose(),
            Normalize(normalize),
            AddGaussian(sigma=0.05),
            Retype()
        ])


    else:
        raise ValueError("Unknown transform mode!")
