"""TESSA model: a benchmark-oriented Evidential Turing Process implementation.

The model follows the classification realization from *Evidential Turing
Processes* (ICLR 2022). ``TESSA`` is the public method name used by this
benchmark; the paper and reference implementation call the method ETP.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Dirichlet, Normal, kl_divergence


class VariationalLinear(nn.Module):
    """Mean-field variational linear layer used by TESSA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        prior_precision: float = 10.0,
        map_prediction: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_precision = prior_precision
        self.map_prediction = map_prediction
        self.bias = nn.Parameter(torch.empty(out_features))
        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_variance = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.normal_(self.weight_mean, mean=0.0, std=std)
        nn.init.normal_(self.weight_log_variance, mean=-9.0, std=0.001)
        nn.init.zeros_(self.bias)

    def kl_divergence(self) -> Tensor:
        log_variance = self.weight_log_variance.clamp(-11, 11)
        return 0.5 * (
            self.prior_precision * (self.weight_mean.square() + log_variance.exp())
            - log_variance
            - 1
            - math.log(self.prior_precision)
        ).sum()

    def forward(self, inputs: Tensor) -> Tensor:
        mean = F.linear(inputs, self.weight_mean, self.bias)
        if self.map_prediction and not self.training:
            return mean
        variance = F.linear(inputs.square(), self.weight_log_variance.clamp(-11, 11).exp())
        return mean + (variance + 1e-8).sqrt() * torch.randn_like(mean)


class TESSA(nn.Module):
    """Evidential classifier with variational weights and external memory."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        dataset_size: int,
        *,
        memory_size: int = 20,
        memory_decay: float = 0.9,
        memory_std: float = 0.01,
        context_size: int = 128,
        prior_precision: float = 10.0,
        reg_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if memory_size <= 0 or context_size <= 0:
            raise ValueError("memory_size and context_size must be positive")
        if not 0 <= memory_decay <= 1:
            raise ValueError("memory_decay must be in [0, 1]")
        if memory_std <= 0 or prior_precision <= 0 or reg_weight < 0:
            raise ValueError("memory_std and prior_precision must be positive; reg_weight non-negative")

        self.backbone = backbone
        self.num_classes = num_classes
        self.dataset_size = dataset_size
        self.memory_decay = memory_decay
        self.memory_std = memory_std
        self.context_size = context_size
        self.reg_weight = reg_weight

        memory = torch.empty(memory_size, num_classes).normal_(0, 0.01).square_()
        self.register_buffer("memory", memory)
        self.predictor = VariationalLinear(
            num_classes * 2,
            num_classes,
            prior_precision=prior_precision,
        )
        self.key = VariationalLinear(
            num_classes,
            num_classes,
            prior_precision=prior_precision,
        )

    def set_reg_weight(self, value: float) -> None:
        self.reg_weight = float(value)

    def _memory_sample(self) -> Tensor:
        if not self.training:
            return self.memory
        return Normal(self.memory, torch.full_like(self.memory, self.memory_std)).rsample()

    def _attention_weights(self, embeddings: Tensor, memory: Tensor) -> Tensor:
        keys = self.key(memory)
        scores = embeddings @ keys.transpose(0, 1) / math.sqrt(self.num_classes)
        return F.softmax(scores, dim=1)

    def _attention(self, embeddings: Tensor) -> Tensor:
        memory = self._memory_sample()
        return self._attention_weights(embeddings, memory) @ memory

    @torch.no_grad()
    def _update_memory(self, embeddings: Tensor, targets: Tensor) -> None:
        embeddings = embeddings[: self.context_size]
        targets = targets[: self.context_size]
        memory = self._memory_sample()
        weights = self._attention_weights(embeddings, memory)
        observations = F.one_hot(targets, self.num_classes) + F.softmax(embeddings, dim=1)
        update = weights.transpose(0, 1) @ observations
        self.memory.mul_(self.memory_decay).add_(update, alpha=1 - self.memory_decay).tanh_()

    def forward(self, inputs: Tensor) -> Tensor:
        embeddings = self.backbone(inputs)
        attention = self._attention(embeddings)
        return self.predictor(torch.cat((embeddings, attention), dim=1)).clamp(max=15)

    def variational_kl(self) -> Tensor:
        return sum(
            layer.kl_divergence()
            for layer in self.modules()
            if isinstance(layer, VariationalLinear)
        )

    def compute_loss(self, inputs: Tensor, targets: Tensor) -> Tensor:
        one_hot = F.one_hot(targets, self.num_classes)
        embeddings = self.backbone(inputs)
        attention = self._attention(embeddings)
        log_concentration = self.predictor(torch.cat((embeddings, attention), dim=1)).clamp(max=15)
        self._update_memory(embeddings, targets)

        concentration = log_concentration.exp()
        strength = concentration.sum(dim=1, keepdim=True)
        fit = (one_hot * (torch.digamma(strength + 1e-8) - torch.digamma(concentration + 1e-8))).sum(1)
        regularization = kl_divergence(Dirichlet(concentration), Dirichlet(attention.exp()))
        return (fit + self.reg_weight * regularization).mean() + self.variational_kl() / self.dataset_size


__all__ = ["TESSA", "VariationalLinear"]
