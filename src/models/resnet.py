from collections.abc import Callable
from typing import Literal

from torch import Tensor, nn
from torch.nn.functional import relu
import torch.nn.functional as F

from src.models.layers.bayesian import BayesConv1d, BayesLinear
from src.models.layers.mc_batch_norm import MCBatchNorm1d
from src.models.layers.packed import PackedConv1d, PackedLinear
from src.models.wrappers.batch_ensemble import BatchEnsemble
from src.models.wrappers.stochastic import StochasticModel

__all__ = ["resnet1d", "batchensemble_resnet1d", "packed_resnet1d",  "bayesian_resnet1d"]

class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        conv1d_layer: type[nn.Module],
        layer_args: dict,
        dropout_rate: float,
        groups: int,
        activation: Callable,
        norm: type[nn.Module],
        conv_bias: bool,
    ) -> None:
        super().__init__()
        self.activation = activation

        self.conv1 = conv1d_layer(
            in_planes, planes,
            kernel_size=3, stride=stride, padding=1,
            groups=groups, bias=conv_bias, **layer_args
        )
        self.norm1 = norm(planes) if norm != nn.Identity else norm()

        self.dropout = nn.Dropout(p=dropout_rate)

        self.conv2 = conv1d_layer(
            planes, planes,
            kernel_size=3, stride=1, padding=1,
            groups=groups, bias=conv_bias, **layer_args
        )
        self.norm2 = norm(planes) if norm != nn.Identity else norm()

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                conv1d_layer(
                    in_planes, planes,
                    kernel_size=1, stride=stride,
                    groups=groups, bias=conv_bias, **layer_args
                ),
                norm(planes) if norm != nn.Identity else norm(),
            )

    def forward(self, x):
        out = self.dropout(self.activation(self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        out += self.shortcut(x)
        return self.activation(out)


class _ResNet1D(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        in_channels,
        num_classes,
        conv1d_layer,
        linear_layer,
        layer_args,
        conv_bias,
        dropout_rate,
        groups,
        in_planes,
        activation,
        norm,
    ):
        super().__init__()
        self.in_planes = in_planes
        self.activation = activation
        if conv1d_layer == PackedConv1d:
            self.conv1 = conv1d_layer(
                in_channels, in_planes,
                kernel_size=7, stride=2, padding=3,
                bias=conv_bias, first=True, **layer_args
            )
        else:
            self.conv1 = conv1d_layer(
                in_channels, in_planes,
                kernel_size=7, stride=2, padding=3,
                bias=conv_bias, **layer_args
            )

        self.norm1 = norm(in_planes) if norm != nn.Identity else norm()
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], 1,
                                       conv1d_layer, layer_args, dropout_rate, groups, activation, norm, conv_bias)
        self.layer2 = self._make_layer(block, in_planes * 2, num_blocks[1], 2,
                                       conv1d_layer, layer_args, dropout_rate, groups, activation, norm, conv_bias)
        self.layer3 = self._make_layer(block, in_planes * 4, num_blocks[2], 2,
                                       conv1d_layer, layer_args, dropout_rate, groups, activation, norm, conv_bias)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)


        if linear_layer == PackedLinear:
            self.fc = linear_layer(in_planes * 4, num_classes, last=True, **layer_args)

        else:
            self.fc = linear_layer(in_planes * 4, num_classes, **layer_args)

    def _make_layer(self, block, planes, blocks, stride,
                    conv1d_layer, layer_args, dropout_rate, groups, activation, norm, conv_bias):
        layers = []
        for i in range(blocks):
            layers.append(block(
                self.in_planes,
                planes,
                stride if i == 0 else 1,
                conv1d_layer,
                layer_args,
                dropout_rate,
                groups,
                activation,
                norm,
                conv_bias,
            ))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        return self.fc(x)
    
    def feats_forward(self, x: Tensor) -> Tensor:
        """返回 fc 之前的特征向量，供对比学习使用。"""
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return self.flatten(x)              # (B, in_planes*4)

def _resnet1d(
    stochastic: bool,
    in_channels: int,
    num_classes: int,
    layer_args: dict,
    num_samples: int = 16,
    conv1d_layer: type[nn.Module] = nn.Conv1d,
    linear_layer: type[nn.Module] = nn.Linear,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.BatchNorm1d,
    groups: int = 1,
    dropout_rate: float = 0.0,
):
    model = _ResNet1D(
        block=_BasicBlock,
        num_blocks=[2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        conv1d_layer=conv1d_layer,
        linear_layer=linear_layer,
        layer_args=layer_args,
        conv_bias=False,
        dropout_rate=dropout_rate,
        groups=groups,
        in_planes=16,
        activation=activation,
        norm=norm,
    )
    if stochastic:
        return StochasticModel(model, num_samples)
    return model

def resnet1d(
    in_channels: int,
    num_classes: int,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.BatchNorm1d,
    groups: int = 1,
    dropout_rate: float = 0.0,
):
    return _resnet1d(
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


def batchensemble_resnet1d(
    in_channels: int,
    num_classes: int,
    num_estimators: int = 4,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.BatchNorm1d,
    groups: int = 1,
    dropout_rate: float = 0.0,
    repeat_training_inputs: bool = False,
):
    model = resnet1d(
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

def packed_resnet1d(
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
    return _resnet1d(
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



def bayesian_resnet1d(
    in_channels: int,
    num_classes: int,
    num_samples: int = 16,
    prior_sigma_1: float | None = None,
    prior_sigma_2: float | None = None,
    prior_pi: float | None = None,
    mu_init: float | None = None,
    sigma_init: float | None = None,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.BatchNorm1d,
    groups: int = 1,
    dropout_rate: float = 0.0,
):
    layer_args = {}
    if prior_sigma_1 is not None:
        layer_args["prior_sigma_1"] = prior_sigma_1
    if prior_sigma_2 is not None:
        layer_args["prior_sigma_2"] = prior_sigma_2
    if prior_pi is not None:
        layer_args["prior_pi"] = prior_pi
    if mu_init is not None:
        layer_args["mu_init"] = mu_init
    if sigma_init is not None:
        layer_args["sigma_init"] = sigma_init

    return _resnet1d(
        stochastic=True,
        num_samples=num_samples,
        in_channels=in_channels,
        num_classes=num_classes,
        conv1d_layer=BayesConv1d,
        linear_layer=BayesLinear,
        layer_args=layer_args,
        activation=activation,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
    )

