"""Calibration-error metric factory without visualization dependencies."""

from typing import Any, Literal

from torchmetrics.classification.calibration_error import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel

from .adaptive_calibration_error import AdaptiveCalibrationError


class CalibrationError:
    """Create a binary or multiclass calibration-error metric."""

    def __new__(  # type: ignore[misc]
        cls,
        task: Literal["binary", "multiclass"],
        adaptive: bool = False,
        num_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        if kwargs.get("n_bins") is not None:
            raise ValueError("`n_bins` does not exist, use `num_bins`.")

        if adaptive:
            return AdaptiveCalibrationError(
                task=task,
                num_bins=num_bins,
                norm=norm,
                num_classes=num_classes,
                ignore_index=ignore_index,
                validate_args=validate_args,
                **kwargs,
            )

        task_type = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {
                "n_bins": num_bins,
                "norm": norm,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task_type == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCalibrationError(**kwargs)
        if not isinstance(num_classes, int):
            raise TypeError(
                "`num_classes` must be an integer for multiclass calibration."
            )
        return MulticlassCalibrationError(num_classes, **kwargs)
