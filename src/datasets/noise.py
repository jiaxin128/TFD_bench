# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
"""Reproducible corruptions for one-dimensional signal evaluation."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


# Five severity levels. Keep only the corruptions that should be evaluated.
DEFAULT_NOISE_PARAMS: dict[str, Sequence[Any]] = {
    "gaussian": [14, 12, 10, 8, 6],
    # "impulse": [(0.001, 3), (0.003, 4), (0.006, 5), (0.01, 6), (0.02, 8)],
    # "clip": [0.95, 0.85, 0.70, 0.55, 0.40],
    # "scale": [0.9, 0.75, 0.6, 0.45, 0.3],
    # "dropout": [0.01, 0.03, 0.06, 0.10, 0.15],
    # "drift": [0.05, 0.1, 0.2, 0.35, 0.5],
}


class AddGaussian:
    """Add fixed-standard-deviation Gaussian noise in a transform pipeline."""

    def __init__(self, sigma: float = 0.01) -> None:
        self.sigma = sigma

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return signal + np.random.normal(0, self.sigma, size=signal.shape)


def _add_gaussian(signal: np.ndarray, snr_db: float, rng) -> np.ndarray:
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=signal.shape).astype(np.float32)
    return signal + noise


def _add_impulse(signal: np.ndarray, probability: float, scale: float, rng) -> np.ndarray:
    output = signal.copy()
    mask = rng.random(signal.shape) < probability
    spikes = rng.normal(0, np.std(signal) * scale, size=signal.shape).astype(np.float32)
    output[mask] += spikes[mask]
    return output


def _add_clip(signal: np.ndarray, quantile: float) -> np.ndarray:
    threshold = np.quantile(np.abs(signal), quantile)
    return np.clip(signal, -threshold, threshold)


def _add_scale(signal: np.ndarray, factor: float) -> np.ndarray:
    return signal * factor


def _add_dropout(signal: np.ndarray, probability: float, rng) -> np.ndarray:
    output = signal.copy()
    output[rng.random(signal.shape) < probability] = 0.0
    return output


def _add_drift(signal: np.ndarray, amplitude_ratio: float, rng) -> np.ndarray:
    length = signal.shape[0]
    frequency = rng.uniform(0.5, 2.0)
    time = np.linspace(0, 1, length)
    drift = (
        amplitude_ratio
        * np.std(signal)
        * np.sin(2 * np.pi * frequency * time)
    ).reshape(-1, 1)
    return (signal + drift).astype(np.float32)


def add_noise(
    signal: np.ndarray,
    noise_type: str,
    severity: int,
    *,
    params: Mapping[str, Sequence[Any]] = DEFAULT_NOISE_PARAMS,
    seed: int = 42,
) -> np.ndarray:
    """Apply one deterministic corruption at severity 1 through 5."""
    if not 1 <= severity <= 5:
        raise ValueError(f"severity must be between 1 and 5, got {severity}.")
    if noise_type not in params:
        raise ValueError(f"Unknown noise type: {noise_type}. Available: {list(params)}")
    if len(params[noise_type]) != 5:
        raise ValueError(f"Noise type {noise_type!r} must define five severity levels.")

    rng = np.random.default_rng(seed)
    value = params[noise_type][severity - 1]
    if noise_type == "gaussian":
        return _add_gaussian(signal, value, rng)
    if noise_type == "impulse":
        probability, scale = value
        return _add_impulse(signal, probability, scale, rng)
    if noise_type == "clip":
        return _add_clip(signal, value)
    if noise_type == "scale":
        return _add_scale(signal, value)
    if noise_type == "dropout":
        return _add_dropout(signal, value, rng)
    if noise_type == "drift":
        return _add_drift(signal, value, rng)
    raise ValueError(f"Noise implementation is missing for {noise_type!r}.")


def build_noisy_df(
    frame: pd.DataFrame,
    noise_type: str,
    severity: int,
    *,
    params: Mapping[str, Sequence[Any]] = DEFAULT_NOISE_PARAMS,
    base_seed: int = 42,
) -> pd.DataFrame:
    """Return a reproducibly corrupted copy of a signal DataFrame."""
    noisy_data = [
        add_noise(
            signal,
            noise_type,
            severity,
            params=params,
            seed=base_seed + index,
        )
        for index, signal in enumerate(frame["data"])
    ]
    result = pd.DataFrame({"data": noisy_data, "label": frame["label"].values})
    if "source" in frame:
        result["source"] = frame["source"].values
    return result


class NoisyEvaluationMixin:
    """Shared noisy ID/OOD dataset construction for signal DataModules."""

    noise_params = DEFAULT_NOISE_PARAMS

    def get_noisy_test_set(self, noise_type: str, severity: int):
        from src.datasets.base_dataset import dataset

        frame = build_noisy_df(
            self.test_df, noise_type, severity, params=self.noise_params
        )
        return dataset(list_data=frame, transform=self.test_transform)

    def get_noisy_ood_set(self, noise_type: str, severity: int):
        from src.datasets.base_dataset import dataset

        if not getattr(self, "eval_ood", False):
            raise RuntimeError("OOD evaluation is disabled for this DataModule.")
        frame = build_noisy_df(
            self.ood_df, noise_type, severity, params=self.noise_params
        )
        return dataset(list_data=frame, transform=self.ood_transform)
