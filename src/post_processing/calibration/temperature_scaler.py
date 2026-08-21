# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
from typing import Literal

import torch
from torch import Tensor, nn

from .scaler import Scaler


class TemperatureScaler(Scaler):
    def __init__(
        self,
        model: nn.Module | None = None,
        init_val: float = 1,
        lr: float = 0.1,
        max_iter: int = 100,
        eps: float = 1e-8,
        min_temperature: float = 0.1,
        max_temperature: float = 10.0,
        device: Literal["cpu", "cuda"] | torch.device | None = None,
    ) -> None:
        """Temperature scaling post-processing for calibrated probabilities.

        Args:
            model (nn.Module): Model to calibrate.
            init_val (float, optional): Initial value for the temperature. Defaults to ``1``.
            lr (float, optional): Learning rate for the optimizer. Defaults to ``0.1``.
            max_iter (int, optional): Maximum number of iterations for the optimizer. Defaults to ``100``.
            eps (float): Small value for stability. Defaults to ``1e-8``.
            device (Optional[Literal["cpu", "cuda"]], optional): Device to use for optimization. Defaults to ``None``.

        References:
            [1] `On calibration of modern neural networks. In ICML 2017
            <https://arxiv.org/abs/1706.04599>`_.
        """
        super().__init__(model=model, lr=lr, max_iter=max_iter, eps=eps, device=device)

        if min_temperature <= 0:
            raise ValueError("min_temperature must be strictly positive.")
        if max_temperature <= min_temperature:
            raise ValueError("max_temperature must be greater than min_temperature.")
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)

        if init_val <= 0:
            raise ValueError(f"Initial temperature value must be positive. Got {init_val}")

        self.set_temperature(init_val)

    def set_temperature(self, val: float) -> None:
        """Set the temperature to a fixed value.

        Args:
            val (float): Temperature value.
        """
        if not self.min_temperature <= val <= self.max_temperature:
            raise ValueError(
                f"Temperature must be in [{self.min_temperature}, "
                f"{self.max_temperature}]. Got {val}."
            )
        # Optimize an unconstrained leaf while exposing a bounded positive
        # temperature. This prevents zero/negative or extremely large values
        # on nearly separable calibration sets.
        unit = (val - self.min_temperature) / (
            self.max_temperature - self.min_temperature
        )
        unit = min(max(unit, self.eps), 1 - self.eps)
        raw = torch.logit(torch.tensor([unit], device=self.device))
        self.raw_temp = nn.Parameter(raw, requires_grad=True)

    def _scale(self, logits: Tensor) -> Tensor:
        return logits / self.temperature[0]

    @property
    def temperature(self) -> list:
        unit = torch.sigmoid(self.raw_temp)
        value = self.min_temperature + (
            self.max_temperature - self.min_temperature
        ) * unit
        return [value]

    @property
    def optimization_parameters(self) -> list[nn.Parameter]:
        return [self.raw_temp]

