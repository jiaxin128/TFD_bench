"""
Plot All — generate all visualization figures from experiment results.

Usage:
    python analysis/visualization/plot_all.py --results results/summary.json --output figures/
    python analysis/visualization/plot_all.py --demo
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('TkAgg')

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.visualization.reliability import plot_reliability_diagram, plot_multi_reliability
from analysis.visualization.uncertainty import plot_uncertainty_distribution, plot_violin_comparison
from analysis.visualization.roc import plot_roc_curves, plot_pr_curves, plot_roc_and_pr
from analysis.visualization.comparison import plot_metric_comparison, plot_metric_heatmap, plot_radar_chart

_PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_results(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def generate_all_plots(results: dict, output_dir: str, dpi: int = 150):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("Generating Visualization Figures")
    print("=" * 60)

    metrics_dict = {}
    for key, stats in results.items():
        method = stats.get("method", key)
        metrics = {k: v["mean"] if isinstance(v, dict) else v
                   for k, v in stats.get("metrics", {}).items()}
        if metrics:
            metrics_dict[method] = metrics

    if metrics_dict:
        for name, fig_fn, kwargs in [
            ("comparison_bar.png",     plot_metric_comparison, {"title": "Method Comparison"}),
            ("comparison_heatmap.png", plot_metric_heatmap,    {"title": "Method Comparison Heatmap"}),
        ]:
            fig = fig_fn(metrics_dict, **kwargs)
            path = os.path.join(output_dir, name)
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")

        if len(list(metrics_dict.values())[0]) >= 3:
            fig = plot_radar_chart(metrics_dict, title="Method Comparison (Radar)")
            path = os.path.join(output_dir, "comparison_radar.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")

    print(f"\nAll figures saved to: {output_dir}")


def generate_demo_plots(output_dir: str, dpi: int = 150):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    print("Generating demo figures...")

    confidences = np.random.beta(5, 2, 1000)
    correctness = np.random.binomial(1, confidences * 0.85)
    fig = plot_reliability_diagram(confidences, correctness, title="Sample Reliability Diagram")
    fig.savefig(os.path.join(output_dir, "demo_reliability.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    id_scores = np.random.beta(2, 5, 500)
    ood_scores = np.random.beta(5, 2, 500)
    fig = plot_uncertainty_distribution(id_scores, ood_scores, title="Sample Uncertainty Distribution")
    fig.savefig(os.path.join(output_dir, "demo_uncertainty.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    labels = np.concatenate([np.zeros(500), np.ones(500)])
    roc_data = {
        "Method A": (labels, np.concatenate([np.random.beta(2, 5, 500), np.random.beta(5, 2, 500)])),
        "Method B": (labels, np.concatenate([np.random.beta(3, 4, 500), np.random.beta(4, 3, 500)])),
    }
    fig = plot_roc_and_pr(roc_data, title="Sample OOD Detection")
    fig.savefig(os.path.join(output_dir, "demo_roc_pr.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    demo = {
        "MaxSoftmax": {"Accuracy": 0.95, "ECE": 0.08, "AUROC": 0.85},
        "MC Dropout":  {"Accuracy": 0.96, "ECE": 0.05, "AUROC": 0.89},
        "EDL":         {"Accuracy": 0.97, "ECE": 0.03, "AUROC": 0.92},
        "Ensemble":    {"Accuracy": 0.98, "ECE": 0.02, "AUROC": 0.94},
    }
    fig = plot_metric_comparison(demo, title="Sample Method Comparison")
    fig.savefig(os.path.join(output_dir, "demo_comparison.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Demo figures saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate all visualization figures")
    parser.add_argument("--results", default=str(_PROJECT_ROOT / "results" / "summary.json"))
    parser.add_argument("--output", default="figures")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        generate_demo_plots(args.output, args.dpi)
    else:
        generate_all_plots(load_results(args.results), args.output, args.dpi)


if __name__ == "__main__":
    main()
