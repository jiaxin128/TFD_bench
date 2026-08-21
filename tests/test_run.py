"""Tests for the benchmark matrix runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

import run as benchmark_run


class RunConfigTests(unittest.TestCase):
    def write_config(self, root: Path, payload: dict) -> Path:
        path = root / "experiment.yaml"
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        return path

    def test_builds_dataset_method_backbone_product(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "methods": ["max_softmax", "edl"],
                "backbones": ["resnet"],
                "datasets": [{"name": "seu", "root": "./data/SEU"}],
                "method_args": {"edl": {"reg_weight": 0.01}},
            }
            path = self.write_config(root, config)
            loaded = benchmark_run.load_config(path)
            commands = benchmark_run.build_commands(path, loaded)

        self.assertEqual(len(commands), 2)
        self.assertEqual(Path(commands[0][1]).stem, "max_softmax")
        self.assertEqual(Path(commands[1][1]).stem, "edl")
        self.assertIn("--reg-weight", commands[1])
        self.assertEqual(commands[1][commands[1].index("--reg-weight") + 1], "0.01")

    def test_rejects_incomplete_dataset(self) -> None:
        config = {
            "methods": ["max_softmax"],
            "backbones": ["resnet"],
            "datasets": [{"name": "seu"}],
        }
        with self.assertRaisesRegex(ValueError, "name and root"):
            benchmark_run.build_commands(Path("invalid.yaml"), config)

    def test_boolean_options_use_argparse_boolean_form(self) -> None:
        command: list[str] = []
        benchmark_run.append_option(command, "eval_noise", False)
        benchmark_run.append_option(command, "overwrite", True)
        self.assertEqual(command, ["--no-eval-noise", "--overwrite"])


if __name__ == "__main__":
    unittest.main()