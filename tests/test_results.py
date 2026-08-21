"""Tests for the public result schema and analysis loaders."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.collect_results import collect_from_results_dir, compute_summary_statistics
from analysis.visualization.io import discover_prediction_runs, finite_ood_score_pair
from src.training.experiment import _build_summary, _write_manifest


class ResultSchemaTests(unittest.TestCase):
    def test_collects_runs_and_computes_seed_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            method_dir = root / "demo" / "resnet" / "max_softmax"
            method_dir.mkdir(parents=True)
            with (method_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["seed", "config", "test/cls/Acc", "test/cal/ECE", "ood/AUROC"],
                )
                writer.writeheader()
                writer.writerows([
                    {"seed": 0, "config": "clean", "test/cls/Acc": 0.8, "test/cal/ECE": 0.1, "ood/AUROC": 0.7},
                    {"seed": 1, "config": "clean", "test/cls/Acc": 1.0, "test/cal/ECE": 0.2, "ood/AUROC": 0.9},
                ])

            records = collect_from_results_dir(str(root))
            summary = compute_summary_statistics(records)

        self.assertEqual(len(records), 2)
        stats = summary["demo/resnet/max_softmax/clean"]
        self.assertEqual(stats["n_runs"], 2)
        self.assertAlmostEqual(stats["metrics"]["test/cls/Acc"]["mean"], 0.9)
        self.assertEqual(stats["metrics"]["ood/AUROC"]["n"], 2)

    def test_tidy_summary_and_manifest_contract(self) -> None:
        frame = pd.DataFrame([
            {"seed": 0, "config": "clean", "test/cls/Acc": 0.8},
            {"seed": 1, "config": "clean", "test/cls/Acc": 1.0},
        ])
        summary = _build_summary(frame)
        self.assertEqual(list(summary.columns), ["config", "metric", "mean", "std", "n"])
        self.assertEqual(int(summary.iloc[0]["n"]), 2)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(dataset="demo", backbone="resnet", output_dir=str(root))
            _write_manifest(root, args, "max_softmax", [0, 1], status="complete")
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["files"]["runs"], "runs.csv")

    def test_prediction_loader_ignores_legacy_object_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pred_dir = root / "demo" / "resnet" / "max_softmax" / "seed0" / "predictions"
            pred_dir.mkdir(parents=True)
            np.savez(
                pred_dir / "clean.npz",
                id_probs=np.array([[0.9, 0.1], [0.2, 0.8]]),
                id_ood_scores=np.array([0.1, np.nan]),
                ood_scores=np.array([0.8, np.inf]),
                ood_criterion=np.array("legacy", dtype=object),
            )
            grouped = discover_prediction_runs(
                root, dataset="demo", backbone="resnet", config="clean"
            )
            arrays = grouped["max_softmax"][0][1]
            id_scores, ood_scores = finite_ood_score_pair(arrays)

        self.assertEqual(id_scores.tolist(), [0.1])
        self.assertEqual(ood_scores.tolist(), [0.8])
        self.assertNotIn("ood_criterion", arrays)


if __name__ == "__main__":
    unittest.main()