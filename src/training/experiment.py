"""Shared multi-run experiment infrastructure for all method entry points."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn, optim
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from src.utils import get_datamodule, get_model, get_trainer_kwargs, load_config

from .classification import ClassificationRoutine
from .utils import TUTrainer
Metrics = dict[str, Any]
Results = dict[str, Metrics]


def add_experiment_args(parser, *, checkpoint: bool = False):
    """Add the repeatability arguments shared by every method script."""
    config = load_config(parser.get_default("config"))
    training = config.get("training", {})
    runner = config.get("runner", {})
    parser.add_argument("--n-runs", type=int, default=training.get("n_runs", 5))
    parser.add_argument("--seeds", type=int, nargs="+", default=training.get("seeds"))
    parser.add_argument("--val-split", type=float, default=training.get("val_split", 0.2))
    parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=runner.get("overwrite", True),
        help="Replace the existing result directory for this dataset/backbone/method",
    )
    if checkpoint:
        parser.add_argument("--ckpt", type=str, default=None)
    return parser


def method_dir(args, method: str) -> Path:
    return Path(args.output_dir) / str(args.dataset) / args.backbone / method


def seed_dir(args, method: str, seed: int) -> Path:
    return method_dir(args, method) / f"seed{seed}"


def make_trainer(
    args,
    run_dir: Path,
    *,
    max_epochs: int | None = None,
    monitor: str | None = "val/cls/NLL",
    mode: str = "min",
    callbacks: Sequence[Any] = (),
    checkpoint: bool = True,
    checkpoint_name: str = "best-{epoch}",
    **trainer_overrides,
) -> TUTrainer:
    """Create a consistently logged trainer and optional best checkpoint."""
    run_dir.mkdir(parents=True, exist_ok=True)
    all_callbacks = [TQDMProgressBar(refresh_rate=1)]
    if checkpoint:
        all_callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir / "ckpt",
                monitor=monitor,
                mode=mode,
                save_top_k=1,
                filename=checkpoint_name,
                verbose=False,
            )
        )
    all_callbacks.extend(callbacks)
    kwargs = get_trainer_kwargs(args)
    kwargs.update(trainer_overrides)
    return TUTrainer(
        **kwargs,
        max_epochs=max_epochs if max_epochs is not None else args.epochs,
        logger=CSVLogger(save_dir=str(run_dir), name="", version="logs"),
        callbacks=all_callbacks,
    )


def load_best_weights(trainer: TUTrainer, routine) -> Path:
    """Load the best ModelCheckpoint weights into the fitted routine in place."""
    checkpoint = next(
        callback
        for callback in trainer.callbacks
        if isinstance(callback, ModelCheckpoint)
    )
    path = Path(checkpoint.best_model_path)
    if not path.exists():
        raise FileNotFoundError(f"Best checkpoint was not created: {path}")
    payload = torch.load(path, map_location="cpu")
    routine.load_state_dict(payload["state_dict"])
    return path


def train_base_classifier(
    args,
    run_dir: Path,
    *,
    model_kwargs: Mapping[str, Any] | None = None,
):
    """Train a deterministic base classifier and restore its best weights."""
    datamodule = get_datamodule(args, val_split=args.val_split, eval_ood=True)
    model = get_model(
        args.backbone,
        datamodule.num_channels,
        datamodule.num_classes,
        **dict(model_kwargs or {}),
    )
    routine = ClassificationRoutine(
        model=model,
        num_classes=datamodule.num_classes,
        loss=nn.CrossEntropyLoss(),
        optim_recipe=optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3),
    )
    trainer = make_trainer(args, run_dir)
    trainer.fit(model=routine, datamodule=datamodule)
    load_best_weights(trainer, routine)
    return trainer, datamodule, model


def train_postprocess_and_evaluate(
    args,
    run_dir: Path,
    build_postprocess: Callable[[Any, Any], Any],
    *,
    ood_criterion: Any | None = None,
) -> Results:
    """Train a base classifier, fit one post-processor, then evaluate it."""
    trainer, datamodule, model = train_base_classifier(args, run_dir)
    postprocess = build_postprocess(model, datamodule)
    if getattr(postprocess, "model", None) is None:
        postprocess.set_model(model)
    postprocess.fit(datamodule.postprocess_dataloader())
    postprocess.fit = lambda *args, **kwargs: None
    routine = ClassificationRoutine(
        model=model,
        num_classes=datamodule.num_classes,
        loss=None,
        eval_ood=True,
        post_processing=postprocess,
        log_post_processing=False,
        ood_criterion=ood_criterion or "msp",
        save_in_csv=True,
    )
    return evaluate(args, trainer, routine, datamodule)


def evaluate(
    args,
    trainer: TUTrainer,
    routine,
    datamodule,
    *,
    ckpt_path: str | Path | None = None,
    on_config: Callable[[str], None] | None = None,
) -> Results:
    """Evaluate clean and configured noisy ID/OOD sets."""
    results: Results = {}
    if on_config:
        on_config("clean")
    results["clean"] = trainer.test(
        model=routine,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
    )[0]

    if not getattr(args, "eval_noise", False):
        return results

    required = ("noise_params", "get_noisy_test_set", "get_noisy_ood_set")
    missing = [name for name in required if not hasattr(datamodule, name)]
    if missing:
        raise NotImplementedError(
            f"{type(datamodule).__name__} does not support noise evaluation "
            f"({', '.join(missing)} missing). Run with --no-eval-noise."
        )

    for noise_type in datamodule.noise_params:
        for severity in range(1, 6):
            config = f"{noise_type}_s{severity}"
            print(f"\n=== {config} ===")
            datamodule.test = datamodule.get_noisy_test_set(noise_type, severity)
            if datamodule.eval_ood:
                datamodule.ood = datamodule.get_noisy_ood_set(noise_type, severity)
            datamodule._noisy_mode = True
            try:
                if on_config:
                    on_config(config)
                results[config] = trainer.test(
                    model=routine,
                    datamodule=datamodule,
                )[0]
            finally:
                datamodule._noisy_mode = False
    return results


def fit_and_evaluate(
    args,
    run_dir: Path,
    build_routine: Callable[[Any], Any],
    *,
    monitor: str = "val/cls/NLL",
    mode: str = "min",
    callbacks: Sequence[Any] = (),
    trainer_overrides: Mapping[str, Any] | None = None,
) -> Results:
    """Run the standard train-best-checkpoint-clean/noise workflow."""
    datamodule = get_datamodule(
        args,
        val_split=args.val_split,
        eval_ood=True,
    )
    routine = build_routine(datamodule)
    trainer = make_trainer(
        args,
        run_dir,
        monitor=monitor,
        mode=mode,
        callbacks=callbacks,
        **dict(trainer_overrides or {}),
    )
    trainer.fit(model=routine, datamodule=datamodule)
    return evaluate(args, trainer, routine, datamodule, ckpt_path="best")


def save_seed_metrics(run_dir: Path, seed: int, results: Mapping[str, Metrics]) -> None:
    rows = [
        {"seed": seed, "config": config, **metrics}
        for config, metrics in results.items()
    ]
    pd.DataFrame(rows).to_csv(run_dir / "metrics.csv", index=False)


def run_repeated(
    args,
    method: str,
    run_once: Callable[[Any, int, Path], Results],
) -> pd.DataFrame:
    """Run all seeds, persist partial results, and produce mean/std summaries."""
    seeds = args.seeds if args.seeds else list(range(args.n_runs))
    out_dir = method_dir(args, method)
    if args.overwrite and out_dir.exists():
        output_root = Path(args.output_dir).resolve()
        target = out_dir.resolve()
        if target == output_root or not target.is_relative_to(output_root):
            raise ValueError(f"Refusing to remove unsafe experiment directory: {target}")
        shutil.rmtree(target)
        print(f"Removed previous results: {target}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"method": method, "seeds": seeds, **vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"\n{'=' * 60}\n  {method} | seed={seed}\n{'=' * 60}")
        seed_everything(seed, workers=True)
        run_dir = seed_dir(args, method, seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        results = run_once(args, seed, run_dir)
        save_seed_metrics(run_dir, seed, results)
        rows.extend(
            {"seed": seed, "config": config, **metrics}
            for config, metrics in results.items()
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "raw_all_seeds.csv", index=False)
    metric_cols = [
        col for col in frame.columns
        if col not in {"seed", "config"} and pd.api.types.is_numeric_dtype(frame[col])
    ]
    if metric_cols:
        frame.groupby("config")[metric_cols].agg(["mean", "std"]).to_csv(
            out_dir / "summary.csv"
        )

    print(f"\nSaved results to {out_dir}")
    clean = frame[frame["config"] == "clean"]
    display = [
        col for col in ("seed", "test/cls/Acc", "test/cal/ECE", "ood/AUROC", "ood/FPR95")
        if col in clean.columns
    ]
    if display:
        print(clean[display].to_string(index=False))
    return frame
