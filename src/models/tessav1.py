"""TESSAv1: TESSA with a prototype-contrastive feature constraint."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Dirichlet, kl_divergence

from src.models.tessa import TESSA


class TESSAv1(TESSA):
    """Keep the TESSA architecture and add class-aware feature regularization."""

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        dataset_size: int,
        *,
        feature_reg_weight: float = 0.05,
        feature_temperature: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(backbone, num_classes, dataset_size, **kwargs)
        if feature_reg_weight < 0:
            raise ValueError("feature_reg_weight must be non-negative")
        if feature_temperature <= 0:
            raise ValueError("feature_temperature must be positive")
        self.feature_reg_weight = feature_reg_weight
        self.feature_temperature = feature_temperature
        self.last_feature_loss = torch.tensor(0.0)

    def _feature_contrastive_loss(self, features: Tensor, targets: Tensor) -> Tensor:
        """Pull features toward same-class slots and repel other-class slots."""
        features = F.normalize(features, dim=-1)
        memory = F.normalize(self.memory.detach(), dim=-1)
        logits = features @ memory.transpose(0, 1) / self.feature_temperature
        positive_mask = self.memory_labels.unsqueeze(0) == targets.unsqueeze(1)
        positive_logits = logits.masked_fill(~positive_mask, float("-inf"))
        return (torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)).mean()

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
        fit = (
            one_hot
            * (torch.digamma(strength + 1e-8) - torch.digamma(concentration + 1e-8))
        ).sum(1)
        prior_concentration = self.memory_prior(attention).clamp(max=15).exp()
        regularization = kl_divergence(
            Dirichlet(concentration),
            Dirichlet(prior_concentration),
        )
        evidential_loss = (fit + self.reg_weight * regularization).mean()
        return (
            evidential_loss
            + self.variational_kl() / self.dataset_size
            + self.feature_reg_weight * feature_reg_scale * feature_loss
        )


__all__ = ["TESSAv1"]
