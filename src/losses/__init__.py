# SPDX-License-Identifier: Apache-2.0
# TFD-Bench modification: adapted for one-dimensional fault-diagnosis benchmarking.
# ruff: noqa: F401
from .bayesian import ELBOLoss, KLDiv
from .classification import (
    BCEWithLogitsLSLoss,
    ConfidencePenaltyLoss,
    ConflictualLoss,
    CrossEntropyMaxSupLoss,
    DECLoss,
    FocalLoss,
)
