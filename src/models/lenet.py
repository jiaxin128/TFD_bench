from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from src.models.layers.bayesian import BayesConv1d, BayesLinear
from src.models.layers.mc_batch_norm import MCBatchNorm1d
from src.models.layers.packed import PackedConv1d, PackedLinear
from src.models.wrappers.batch_ensemble import BatchEnsemble
from src.models.wrappers.stochastic import StochasticModel

__all__ = ["lenet1d", "batchensemble_lenet1d", "packed_lenet1d", "bayesian_lenet1d"]


class _LeNet1D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_classes: int,
            linear_layer: type[nn.Module],
            conv1d_layer: type[nn.Module],
            layer_args: dict,
            activation: Callable,
            norm: type[nn.Module],
            groups: int,
            dropout_rate: float,
    ) -> None:
        super().__init__()
        self.activation = activation

        batchnorm = False
        if norm == nn.Identity:
            self.norm1 = norm()
            self.norm2 = norm()
        elif norm == nn.BatchNorm1d or (isinstance(norm, partial) and norm.func == MCBatchNorm1d):
            batchnorm = True
        else:
            raise ValueError("norm must be nn.Identity or nn.BatchNorm1d.")

        self.dropout_rate = dropout_rate
        if conv1d_layer == PackedConv1d:
            self.conv1 = conv1d_layer(in_channels, 6, kernel_size=5, groups=groups, first=True, **layer_args)
        else:
            self.conv1 = conv1d_layer(in_channels, 6, kernel_size=5, groups=groups, **layer_args)

        if batchnorm:
            self.norm1 = norm(6)

        self.conv_dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = conv1d_layer(6, 16, kernel_size=5, groups=groups, **layer_args)
        if batchnorm:
            self.norm2 = norm(16)

        self.pooling = nn.AdaptiveAvgPool1d(4)  # => output length = 4
        self.fc1 = linear_layer(64, 120, **layer_args)
        self.fc_dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = linear_layer(120, 84, **layer_args)
        self.last_fc_dropout = nn.Dropout(p=dropout_rate)
        if linear_layer == PackedLinear:
            self.fc3 = linear_layer(84, num_classes, last=True, **layer_args)
        else:
            self.fc3 = linear_layer(84, num_classes, **layer_args)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, L)
        out = self.conv_dropout(self.activation(self.norm1(self.conv1(x))))
        out = F.max_pool1d(out, kernel_size=2)

        out = self.conv_dropout(self.activation(self.norm2(self.conv2(out))))
        out = F.max_pool1d(out, kernel_size=2)

        out = self.pooling(out)  # -> (N, 16, 4)
        out = torch.flatten(out, 1)  # (N, 64)

        out = self.fc_dropout(self.activation(self.fc1(out)))
        out = self.last_fc_dropout(self.activation(self.fc2(out)))
        return self.fc3(out)


def _lenet1d(
        stochastic: bool,
        in_channels: int,
        num_classes: int,
        layer_args: dict,
        num_samples: int = 16,
        linear_layer: type[nn.Module] = nn.Linear,
        conv1d_layer: type[nn.Module] = nn.Conv1d,
        activation: Callable = nn.ReLU,
        norm: type[nn.Module] = nn.Identity,
        groups: int = 1,
        dropout_rate: float = 0.0,
) -> _LeNet1D | StochasticModel:
    model = _LeNet1D(
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=linear_layer,
        conv1d_layer=conv1d_layer,
        activation=activation,
        norm=norm,
        groups=groups,
        layer_args=layer_args,
        dropout_rate=dropout_rate,
    )
    if stochastic:
        return StochasticModel(model, num_samples)
    return model


def lenet1d(
        in_channels: int,
        num_classes: int,
        activation: Callable = F.relu,
        norm: type[nn.Module] = nn.Identity,
        groups: int = 1,
        dropout_rate: float = 0.0,
) -> _LeNet1D:
    return _lenet1d(
        stochastic=False,
        in_channels=in_channels,
        num_classes=num_classes,
        conv1d_layer=nn.Conv1d,
        linear_layer=nn.Linear,
        layer_args={},
        activation=activation,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
    )


def batchensemble_lenet1d(
        in_channels: int,
        num_classes: int,
        num_estimators: int = 4,
        activation: Callable = F.relu,
        norm: type[nn.Module] = nn.BatchNorm1d,
        groups: int = 1,
        dropout_rate: float = 0.0,
        repeat_training_inputs: bool = False,
):
    model = lenet1d(
        in_channels=in_channels,
        num_classes=num_classes,
        activation=activation,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
    )
    return BatchEnsemble(
        model=model,
        num_estimators=num_estimators,
        repeat_training_inputs=repeat_training_inputs,
        convert_layers=True,
    )


def packed_lenet1d(
        in_channels: int,
        num_classes: int,
        num_estimators: int = 4,
        alpha: float = 2,
        gamma: float = 1,
        activation: Callable = F.relu,
        norm: type[nn.Module] = nn.Identity,
        groups: int = 1,
        dropout_rate: float = 0.0,
):
    return _lenet1d(
        stochastic=False,
        in_channels=in_channels,
        num_classes=num_classes,
        conv1d_layer=PackedConv1d,
        linear_layer=PackedLinear,
        norm=norm,
        layer_args={
            "num_estimators": num_estimators,
            "alpha": alpha,
            "gamma": gamma,
        },
        activation=activation,
        groups=groups,
        dropout_rate=dropout_rate,
    )


def bayesian_lenet1d(
        in_channels: int,
        num_classes: int,
        num_samples: int = 16,
        prior_sigma_1: float | None = None,
        prior_sigma_2: float | None = None,
        prior_pi: float | None = None,
        mu_init: float | None = None,
        sigma_init: float | None = None,
        activation: Callable = F.relu,
        norm: type[nn.Module] = nn.Identity,
        groups: int = 1,
        dropout_rate: float = 0.0,
) -> StochasticModel:
    layers_args = {}
    if prior_sigma_1 is not None:
        layers_args["prior_sigma_1"] = prior_sigma_1
    if prior_sigma_2 is not None:
        layers_args["prior_sigma_2"] = prior_sigma_2
    if prior_pi is not None:
        layers_args["prior_pi"] = prior_pi
    if mu_init is not None:
        layers_args["mu_init"] = mu_init
    if sigma_init is not None:
        layers_args["sigma_init"] = sigma_init

    return _lenet1d(
        stochastic=True,
        num_samples=num_samples,
        in_channels=in_channels,
        num_classes=num_classes,
        conv1d_layer=BayesConv1d,
        linear_layer=BayesLinear,
        norm=norm,
        layer_args=layers_args,
        activation=activation,
        groups=groups,
        dropout_rate=dropout_rate,
    )
