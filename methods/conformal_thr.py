import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.post_processing import ConformalClsTHR
from src.training.experiment import add_experiment_args, run_repeated, train_postprocess_and_evaluate
from src.utils import add_common_args, load_gpu_config, print_gpu_config

METHOD_NAME = "conformal_thr"


def run_once(args, seed, run_dir):
    def build(model, dm):
        return ConformalClsTHR(
            alpha=args.alpha, model=model, enable_ts=args.enable_ts,
            device=next(model.parameters()).device,
        )
    return train_postprocess_and_evaluate(args, run_dir, build, ood_criterion="post_processing")


def run(args):
    return run_repeated(args, METHOD_NAME, run_once)


if __name__ == "__main__":
    parser = add_experiment_args(add_common_args(argparse.ArgumentParser()))
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--enable-ts", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print_gpu_config(load_gpu_config(args))
    run(args)
