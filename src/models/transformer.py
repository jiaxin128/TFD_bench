"""
Transformer for 1D Time-Series Fault Diagnosis

Architecture:
1. Conv1d Stem: Adds local feature extraction before Transformer layers (like Conformer/ConvViT)
2. Sinusoidal Positional Encoding: More stable than learnable PE on small datasets
3. Stochastic Depth: Regularization for small datasets
4. Optional Patch Embedding: Reduces sequence length for efficiency
"""

import math
from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.models.layers.bayesian import BayesLinear
from src.models.layers.packed import PackedLinear
from src.models.wrappers.batch_ensemble import BatchEnsemble
from src.models.wrappers.stochastic import StochasticModel

__all__ = [
    "transformer1d",
    "batchensemble_transformer1d",
    "packed_transformer1d",
    "bayesian_transformer1d",
]


# =============================================================================
# Sinusoidal Positional Encoding (Fixed, not learnable)
# =============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, L, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# =============================================================================
# Conv1d Stem: Local feature extraction before Transformer
# =============================================================================
class ConvStem(nn.Module):
    """Convolutional stem for local feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        pool_kernel: int = 3,
        pool_stride: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride, padding=pool_kernel // 2)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, L)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x  # (N, out_channels, L')


# =============================================================================
# Stochastic Depth (Drop Path) for regularization
# =============================================================================
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# =============================================================================
# Transformer Encoder Block (Pre-LN with DropPath)
# =============================================================================
class _FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float,
        activation: Callable,
        linear_layer: type[nn.Module],
        layer_args: dict,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.fc1 = linear_layer(d_model, d_ff, **layer_args)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = linear_layer(d_ff, d_model, **layer_args)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class _TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer block with DropPath."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout_rate: float,
        drop_path_rate: float,
        activation: Callable,
        norm: type[nn.Module],
        linear_layer: type[nn.Module],
        layer_args: dict,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = norm(d_model) if norm != nn.Identity else norm()
        self.norm2 = norm(d_model) if norm != nn.Identity else norm()

        self.q_proj = linear_layer(d_model, d_model, **layer_args)
        self.k_proj = linear_layer(d_model, d_model, **layer_args)
        self.v_proj = linear_layer(d_model, d_model, **layer_args)
        self.out_proj = linear_layer(d_model, d_model, **layer_args)

        self.attn_dropout = nn.Dropout(p=dropout_rate)
        self.resid_dropout = nn.Dropout(p=dropout_rate)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()

        self.ffn = _FeedForward(
            d_model=d_model,
            d_ff=d_ff,
            dropout_rate=dropout_rate,
            activation=activation,
            linear_layer=linear_layer,
            layer_args=layer_args,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        N, L, _ = x.shape
        return x.view(N, L, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        N, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(N, L, H * D)

    def forward(self, x: Tensor) -> Tensor:
        h = self.norm1(x)

        q = self._split_heads(self.q_proj(h))
        k = self._split_heads(self.k_proj(h))
        v = self._split_heads(self.v_proj(h))

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = self._merge_heads(out)
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        x = x + self.drop_path(out)

        h2 = self.norm2(x)
        out2 = self.ffn(h2)
        out2 = self.resid_dropout(out2)
        x = x + self.drop_path(out2)

        return x


# =============================================================================
# Main Transformer Model
# =============================================================================
class _Transformer1D(nn.Module):
    """Improved Transformer for 1D signals with Conv Stem and Sinusoidal PE."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int,
        n_heads: int,
        depth: int,
        d_ff: int,
        dropout_rate: float,
        drop_path_rate: float,
        activation: Callable,
        norm: type[nn.Module],
        linear_layer: type[nn.Module],
        layer_args: dict,
        use_conv_stem: bool = True,
        conv_stem_channels: int = 64,
        pooling: str = "mean",
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'.")

        self.pooling = pooling
        self.d_model = d_model
        self.use_conv_stem = use_conv_stem

        # 1) Conv Stem (optional but recommended)
        if use_conv_stem:
            self.conv_stem = ConvStem(in_channels, conv_stem_channels)
            self.in_proj = nn.Linear(conv_stem_channels, d_model)
        else:
            self.conv_stem = None
            self.in_proj = linear_layer(in_channels, d_model, **layer_args)

        # 2) Sinusoidal Positional Encoding (fixed)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout_rate)

        # 3) [CLS] token (optional)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        # 4) Transformer Blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay
        self.blocks = nn.Sequential(
            *[
                _TransformerEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout_rate=dropout_rate,
                    drop_path_rate=dpr[i],
                    activation=activation,
                    norm=norm,
                    linear_layer=linear_layer,
                    layer_args=layer_args,
                )
                for i in range(depth)
            ]
        )

        self.final_norm = norm(d_model) if norm != nn.Identity else norm()

        # 5) Classification head
        if linear_layer == PackedLinear:
            self.head = linear_layer(d_model, num_classes, last=True, **layer_args)
        else:
            self.head = linear_layer(d_model, num_classes, **layer_args)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, L)

        if self.use_conv_stem:
            x = self.conv_stem(x)  # (N, conv_stem_channels, L')
            x = x.transpose(1, 2)  # (N, L', conv_stem_channels)
            x = self.in_proj(x)    # (N, L', d_model)
        else:
            x = x.transpose(1, 2)  # (N, L, C)
            x = self.in_proj(x)    # (N, L, d_model)

        N, L, _ = x.shape

        if self.pooling == "cls":
            cls = self.cls_token.expand(N, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.pos_enc(x)
        x = self.blocks(x)
        x = self.final_norm(x)

        if self.pooling == "cls":
            feat = x[:, 0, :]
        else:
            feat = x.mean(dim=1)

        return self.head(feat)


# =============================================================================
# Factory Functions
# =============================================================================
def _transformer1d(
    stochastic: bool,
    in_channels: int,
    num_classes: int,
    layer_args: dict,
    num_samples: int = 16,
    linear_layer: type[nn.Module] = nn.Linear,
    activation: Callable = F.gelu,
    norm: type[nn.Module] = nn.LayerNorm,
    dropout_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    d_model: int = 128,
    n_heads: int = 4,
    depth: int = 4,
    d_ff: int = 256,
    use_conv_stem: bool = True,
    conv_stem_channels: int = 64,
    pooling: str = "mean",
    max_len: int = 4096,
) -> _Transformer1D | StochasticModel:
    model = _Transformer1D(
        in_channels=in_channels,
        num_classes=num_classes,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        d_ff=d_ff,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        activation=activation,
        norm=norm,
        linear_layer=linear_layer,
        layer_args=layer_args,
        use_conv_stem=use_conv_stem,
        conv_stem_channels=conv_stem_channels,
        pooling=pooling,
        max_len=max_len,
    )
    if stochastic:
        return StochasticModel(model, num_samples)
    return model


def transformer1d(
    in_channels: int,
    num_classes: int,
    activation: Callable = F.gelu,
    norm: type[nn.Module] = nn.LayerNorm,
    dropout_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    d_model: int = 128,
    n_heads: int = 4,
    depth: int = 4,
    d_ff: int = 256,
    use_conv_stem: bool = True,
    conv_stem_channels: int = 64,
    pooling: str = "mean",
    max_len: int = 4096,
) -> _Transformer1D:
    """Improved Transformer1D with Conv Stem and Sinusoidal PE."""
    return _transformer1d(
        stochastic=False,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=nn.Linear,
        layer_args={},
        activation=activation,
        norm=norm,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        d_ff=d_ff,
        use_conv_stem=use_conv_stem,
        conv_stem_channels=conv_stem_channels,
        pooling=pooling,
        max_len=max_len,
    )


def packed_transformer1d(
    in_channels: int,
    num_classes: int,
    num_estimators: int = 4,
    alpha: float = 2,
    gamma: float = 1,
    activation: Callable = F.gelu,
    norm: type[nn.Module] = nn.LayerNorm,
    dropout_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    d_model: int = 128,
    n_heads: int = 4,
    depth: int = 4,
    d_ff: int = 256,
    use_conv_stem: bool = True,
    conv_stem_channels: int = 64,
    pooling: str = "mean",
    max_len: int = 4096,
):
    """Packed Ensemble version of Transformer1D."""
    return _transformer1d(
        stochastic=False,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=PackedLinear,
        layer_args={
            "num_estimators": num_estimators,
            "alpha": alpha,
            "gamma": gamma,
        },
        activation=activation,
        norm=norm,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        d_ff=d_ff,
        use_conv_stem=use_conv_stem,
        conv_stem_channels=conv_stem_channels,
        pooling=pooling,
        max_len=max_len,
    )


def batchensemble_transformer1d(
    in_channels: int,
    num_classes: int,
    num_estimators: int = 4,
    activation: Callable = F.gelu,
    norm: type[nn.Module] = nn.LayerNorm,
    dropout_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    d_model: int = 128,
    n_heads: int = 4,
    depth: int = 4,
    d_ff: int = 256,
    use_conv_stem: bool = True,
    conv_stem_channels: int = 64,
    pooling: str = "mean",
    max_len: int = 4096,
    repeat_training_inputs: bool = False,
):
    """Batch Ensemble version of Transformer1D."""
    model = transformer1d(
        in_channels=in_channels,
        num_classes=num_classes,
        activation=activation,
        norm=norm,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        d_ff=d_ff,
        use_conv_stem=use_conv_stem,
        conv_stem_channels=conv_stem_channels,
        pooling=pooling,
        max_len=max_len,
    )
    return BatchEnsemble(
        model=model,
        num_estimators=num_estimators,
        repeat_training_inputs=repeat_training_inputs,
        convert_layers=True,
    )


def bayesian_transformer1d(
    in_channels: int,
    num_classes: int,
    num_samples: int = 16,
    prior_sigma_1: float | None = None,
    prior_sigma_2: float | None = None,
    prior_pi: float | None = None,
    mu_init: float | None = None,
    sigma_init: float | None = None,
    activation: Callable = F.gelu,
    norm: type[nn.Module] = nn.LayerNorm,
    dropout_rate: float = 0.1,
    drop_path_rate: float = 0.1,
    d_model: int = 128,
    n_heads: int = 4,
    depth: int = 4,
    d_ff: int = 256,
    use_conv_stem: bool = True,
    conv_stem_channels: int = 64,
    pooling: str = "mean",
    max_len: int = 4096,
) -> StochasticModel:
    """Bayesian version of Transformer1D."""
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

    return _transformer1d(
        stochastic=True,
        num_samples=num_samples,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=BayesLinear,
        layer_args=layer_args,
        activation=activation,
        norm=norm,
        dropout_rate=dropout_rate,
        drop_path_rate=drop_path_rate,
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        d_ff=d_ff,
        use_conv_stem=use_conv_stem,
        conv_stem_channels=conv_stem_channels,
        pooling=pooling,
        max_len=max_len,
    )
