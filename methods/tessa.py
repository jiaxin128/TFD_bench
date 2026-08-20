import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from lightning.pytorch.utilities.types import STEP_OUTPUT

from src.metrics.ood import OODCriterionInputType, TUOODCriterion
from src.models.tessa import TESSA
from src.training import ClassificationRoutine
from src.training.experiment import add_experiment_args, evaluate, make_trainer, run_repeated
from src.utils import add_common_args, get_datamodule, get_model, load_gpu_config, print_gpu_config

METHOD_NAME = "tessa"


class PredictiveEntropyCriterion(TUOODCriterion):
    """Predictive entropy used for OOD detection in the ETP paper."""

    input_type = OODCriterionInputType.LOGIT
    single_only = True

    def forward(self, log_concentration: Tensor) -> Tensor:
        if log_concentration.ndim == 3:
            log_concentration = log_concentration.squeeze(1)
        probabilities = F.softmax(log_concentration, dim=-1)
        return torch.special.entr(probabilities).sum(dim=-1)


class _TrainingLoss(nn.Module):
    def forward(self, *args, **kwargs):
        raise RuntimeError("TESSARoutine computes its loss through the model")


class TESSARoutine(ClassificationRoutine):
    def __init__(self, model: TESSA, **kwargs):
        kwargs.setdefault("ood_criterion", PredictiveEntropyCriterion())
        super().__init__(model=model, loss=_TrainingLoss(), **kwargs)
        self.tessa = model

    def training_step(self, batch: tuple[Tensor, Tensor], *args, **kwargs) -> STEP_OUTPUT:
        inputs, targets = self.format_batch_fn(self._apply_mixup(batch))
        loss = self.tessa.compute_loss(inputs, targets)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log(
            "train/memory_diversity",
            self.tessa.memory_diversity(),
            prog_bar=False,
            logger=True,
        )


def run_once(args, seed, run_dir):
    datamodule = get_datamodule(args, val_split=args.val_split, eval_ood=True)
    datamodule.setup("fit")
    backbone = get_model(args.backbone, datamodule.num_channels, datamodule.num_classes)
    model = TESSA(
        backbone,
        datamodule.num_classes,
        len(datamodule.train_dataloader().dataset),
        memory_size=args.memory_size,
        memory_decay=args.memory_decay,
        memory_std=args.memory_std,
        context_size=args.context_size,
        prior_precision=args.prior_precision,
        reg_weight=args.reg_weight,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    routine = TESSARoutine(
        model=model,
        num_classes=datamodule.num_classes,
        optim_recipe={
            "optimizer": optimizer,
            "lr_scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs),
        },
        eval_ood=True,
        save_in_csv=True,
    )
    trainer = make_trainer(
        args,
        run_dir,
        gradient_clip_val=args.grad_clip,
    )
    trainer.fit(model=routine, datamodule=datamodule)
    return evaluate(args, trainer, routine, datamodule, ckpt_path="best")


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--reg-weight", type=float, default=1e-3)
    parser.add_argument("--memory-size", type=int, default=20)
    parser.add_argument("--memory-decay", type=float, default=0.99)
    parser.add_argument("--memory-std", type=float, default=0.1)
    parser.add_argument("--context-size", type=int, default=50)
    parser.add_argument("--prior-precision", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
