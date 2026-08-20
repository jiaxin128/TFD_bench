import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import Tensor, optim

from methods.tessa import TESSARoutine
from src.models.tessav1 import TESSAv1
from src.training.experiment import add_experiment_args, evaluate, make_trainer, run_repeated
from src.utils import add_common_args, get_datamodule, get_model, load_gpu_config, print_gpu_config

METHOD_NAME = "tessav1"


class TESSAv1Routine(TESSARoutine):
    def __init__(self, model: TESSAv1, feature_warmup_epochs: int = 10, **kwargs):
        if feature_warmup_epochs <= 0:
            raise ValueError("feature_warmup_epochs must be positive")
        super().__init__(model=model, **kwargs)
        self.tessav1 = model
        self.feature_warmup_epochs = feature_warmup_epochs

    def training_step(self, batch: tuple[Tensor, Tensor], *args, **kwargs) -> STEP_OUTPUT:
        inputs, targets = self.format_batch_fn(self._apply_mixup(batch))
        feature_scale = min((self.current_epoch + 1) / self.feature_warmup_epochs, 1.0)
        loss = self.tessav1.compute_loss(inputs, targets, feature_reg_scale=feature_scale)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train/feature_loss", self.tessav1.last_feature_loss, logger=True)
        self.log("train/feature_scale", feature_scale, logger=True)
        return loss


def run_once(args, seed, run_dir):
    datamodule = get_datamodule(args, val_split=args.val_split, eval_ood=True)
    datamodule.setup("fit")
    backbone = get_model(args.backbone, datamodule.num_channels, datamodule.num_classes)
    model = TESSAv1(
        backbone,
        datamodule.num_classes,
        len(datamodule.train_dataloader().dataset),
        memory_size=args.memory_size,
        memory_decay=args.memory_decay,
        memory_std=args.memory_std,
        context_size=args.context_size,
        prior_precision=args.prior_precision,
        reg_weight=args.reg_weight,
        feature_reg_weight=args.feature_reg_weight,
        feature_temperature=args.feature_temperature,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    routine = TESSAv1Routine(
        model=model,
        num_classes=datamodule.num_classes,
        feature_warmup_epochs=args.feature_warmup_epochs,
        optim_recipe={
            "optimizer": optimizer,
            "lr_scheduler": optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs),
        },
        eval_ood=True,
        save_in_csv=True,
    )
    trainer = make_trainer(args, run_dir, gradient_clip_val=args.grad_clip)
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
    parser.add_argument("--feature-reg-weight", type=float, default=0.05)
    parser.add_argument("--feature-temperature", type=float, default=0.1)
    parser.add_argument("--feature-warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
