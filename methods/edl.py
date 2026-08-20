import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

import torch.nn.functional as F
from torch import nn, optim

from src.losses import DECLoss
from src.metrics.ood import EvidentialCriterion
from src.training import ClassificationRoutine
from src.training.experiment import add_experiment_args, fit_and_evaluate, run_repeated
from src.utils import add_common_args, get_model, load_gpu_config, print_gpu_config


class EvidenceWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, inputs):
        return F.softplus(self.backbone(inputs))


METHOD_NAME = "edl"


def run_once(args, seed, run_dir):
    def build(dm):
        model = EvidenceWrapper(get_model(args.backbone, dm.num_channels, dm.num_classes))
        return ClassificationRoutine(
            model=model, num_classes=dm.num_classes,
            loss=DECLoss(reg_weight=args.reg_weight, loss_type=args.loss_type),
            optim_recipe=optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3),
            eval_ood=True, ood_criterion=EvidentialCriterion(), save_in_csv=True,
        )
    return fit_and_evaluate(args, run_dir, build, monitor="val/cls/Acc", mode="max")


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--reg-weight", type=float, default=0.5)
    parser.add_argument("--loss-type", choices=["digamma", "log", "mse"], default="digamma")
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
