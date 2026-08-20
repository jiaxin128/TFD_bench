"""TESSA model: a feature-memory evidential classifier.

The variational and evidential components are inspired by *Evidential Turing
Processes* (ICLR 2022). Unlike the reference ETP implementation, TESSA stores
class-aware backbone features in its external memory.
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
    """Evidential classifier whose external memory stores backbone features."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        dataset_size: int,
        *,
        memory_size: int = 20,
        memory_decay: float = 0.99,
        memory_std: float = 0.1,
        context_size: int = 50,
        prior_precision: float = 10.0,
        reg_weight: float = 1e-3,
        feature_reg_weight: float = 0.05,
        feature_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if memory_size < num_classes or context_size < 3:
            raise ValueError("memory_size must cover every class and context_size must be at least 3")
        if not 0 <= memory_decay <= 1:
            raise ValueError("memory_decay must be in [0, 1]")
        if memory_std <= 0 or prior_precision <= 0 or feature_temperature <= 0:
            raise ValueError("memory_std, prior_precision and feature_temperature must be positive")
        if reg_weight < 0 or feature_reg_weight < 0:
            raise ValueError("regularization weights must be non-negative")

        if not hasattr(backbone, "feats_forward"):
            raise TypeError("TESSA requires a backbone with feats_forward()")
        classifier = getattr(backbone, "fc", None)
        if not isinstance(classifier, nn.Linear):
            raise TypeError("TESSA currently requires a backbone with an nn.Linear fc head")

        self.feature_dim = classifier.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.num_classes = num_classes
        self.dataset_size = dataset_size
        self.memory_decay = memory_decay
        self.memory_std = memory_std
        self.context_size = context_size
        self.reg_weight = reg_weight
        self.feature_reg_weight = feature_reg_weight
        self.feature_temperature = feature_temperature
        self.last_feature_loss = torch.tensor(0.0)

        memory = F.normalize(torch.randn(memory_size, self.feature_dim), dim=-1)
        self.register_buffer("memory", memory)
        self.register_buffer("memory_labels", torch.arange(memory_size) % num_classes)
        self.feature_projection = VariationalLinear(
            self.feature_dim,
            self.feature_dim,
            prior_precision=prior_precision,
        )
        self.key = nn.Linear(self.feature_dim, self.feature_dim)
        self.predictor = nn.Linear(self.feature_dim * 2, num_classes)
        self.memory_prior = nn.Linear(self.feature_dim, num_classes)

    def _memory_sample(self) -> Tensor:
        return Normal(self.memory, torch.full_like(self.memory, self.memory_std)).rsample()

    def _attention_weights(self, embeddings: Tensor, memory: Tensor) -> Tensor:
        keys = self.key(memory)
        scores = embeddings @ keys.transpose(0, 1) / math.sqrt(self.feature_dim)
        return F.softmax(scores, dim=1)

    def _attention(self, embeddings: Tensor) -> Tensor:
        memory = self._memory_sample()
        return self._attention_weights(embeddings, memory) @ memory

    @torch.no_grad()
    def _update_memory(self, embeddings: Tensor, targets: Tensor) -> None:
        max_context = min(self.context_size, embeddings.size(0))
        context = int(torch.randint(3, max_context + 1, ()).item()) if max_context > 3 else max_context
        embeddings = F.normalize(embeddings[:context], dim=-1)
        targets = targets[:context]

        for class_index in targets.unique():
            class_mask = targets == class_index
            class_features = embeddings[class_mask]
            slot_indices = torch.where(self.memory_labels == class_index)[0]
            slots = F.normalize(self.memory[slot_indices], dim=-1)
            assignments = (class_features @ slots.transpose(0, 1)).argmax(dim=1)
            for local_slot in assignments.unique():
                centroid = class_features[assignments == local_slot].mean(dim=0)
                slot_index = slot_indices[local_slot]
                updated = (
                    self.memory_decay * self.memory[slot_index]
                    + (1 - self.memory_decay) * centroid
                )
                self.memory[slot_index].copy_(F.normalize(updated, dim=0))

    def _features(self, inputs: Tensor) -> Tensor:
        raw_features = self.backbone.feats_forward(inputs)
        return self.feature_projection(raw_features)

    def _log_concentration(self, features: Tensor, attention: Tensor) -> Tensor:
        return self.predictor(torch.cat((features, attention), dim=1)).clamp(max=15)

    def _feature_contrastive_loss(self, features: Tensor, targets: Tensor) -> Tensor:
        """Pull features toward same-class slots and repel other-class slots."""
        features = F.normalize(features, dim=-1)
        memory = F.normalize(self.memory.detach(), dim=-1)
        logits = features @ memory.transpose(0, 1) / self.feature_temperature
        positive_mask = self.memory_labels.unsqueeze(0) == targets.unsqueeze(1)
        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()

    def forward(self, inputs: Tensor) -> Tensor:
        features = self._features(inputs)
        attention = self._attention(features)
        return self._log_concentration(features, attention)

    def variational_kl(self) -> Tensor:
        return sum(
            layer.kl_divergence()
            for layer in self.modules()
            if isinstance(layer, VariationalLinear)
        )

    @torch.no_grad()
    def memory_diversity(self) -> Tensor:
        """Mean pairwise feature-space distance between memory slots."""
        if self.memory.size(0) < 2:
            return self.memory.new_zeros(())
        return torch.pdist(self.memory).mean()

    def compute_loss(
        self,
        inputs: Tensor,
        targets: Tensor,
        *,
        feature_reg_scale: float = 1.0,
    ) -> Tensor:
        one_hot = F.one_hot(targets, self.num_classes)
        features = self._features(inputs)
        attention = self._attention(features)
        log_concentration = self._log_concentration(features, attention)
        feature_loss = self._feature_contrastive_loss(features, targets)
        self.last_feature_loss = feature_loss.detach()
        self._update_memory(features, targets)

        concentration = log_concentration.exp()
        strength = concentration.sum(dim=1, keepdim=True)
        fit = (one_hot * (torch.digamma(strength + 1e-8) - torch.digamma(concentration + 1e-8))).sum(1)
        prior_concentration = self.memory_prior(attention).clamp(max=15).exp()
        regularization = kl_divergence(Dirichlet(concentration), Dirichlet(prior_concentration))
        evidential_loss = (fit + self.reg_weight * regularization).mean()
        return (
            evidential_loss
            + self.variational_kl() / self.dataset_size
            + self.feature_reg_weight * feature_reg_scale * feature_loss
        )


__all__ = ["TESSA", "VariationalLinear"]
