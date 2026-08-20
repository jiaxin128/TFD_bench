"""
Comparison Charts / 对比图表

Multi-method comparison bar charts and heatmaps.
多方法对比柱状图和热力图。

Usage / 使用方法:
    from scripts.visualization import comparison
    fig = comparison.plot_metric_comparison(results_dict)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import matplotlib


def plot_metric_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Method Comparison",
    figsize: Tuple[int, int] = None,
    highlight_best: bool = True,
    higher_better: Dict[str, bool] = None,
) -> plt.Figure:
    """
    Plot grouped bar chart comparing methods across metrics.
    绘制分组柱状图对比各方法的指标。
    
    Args:
        results_dict: Dict of {method_name: {metric_name: value}}
        metrics: List of metric names to plot (auto-detect if None)
        title: Plot title
        figsize: Figure size
        highlight_best: Highlight best value for each metric
        higher_better: Dict of {metric_name: True/False}
    
    Returns:
        matplotlib Figure
    """
    # Get methods and metrics
    methods = list(results_dict.keys())
    if metrics is None:
        metrics = list(list(results_dict.values())[0].keys())
    
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    if figsize is None:
        figsize = (max(8, n_methods * 1.5), 5)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Default higher_better
    if higher_better is None:
        higher_better = {m: not any(x in m.lower() for x in ['loss', 'error', 'fpr', 'nll', 'ece']) 
                        for m in metrics}
    
    # Prepare data
    x = np.arange(n_methods)
    width = 0.8 / n_metrics
    colors = plt.cm.tab10(np.linspace(0, 1, n_metrics))
    
    # Find best values
    best_values = {}
    for metric in metrics:
        values = [results_dict[m].get(metric, 0) for m in methods]
        if higher_better.get(metric, True):
            best_values[metric] = max(values)
        else:
            best_values[metric] = min(values)
    
    # Plot bars
    for i, metric in enumerate(metrics):
        values = [results_dict[m].get(metric, 0) for m in methods]
        offset = (i - n_metrics/2 + 0.5) * width
        
        bars = ax.bar(x + offset, values, width, label=metric, color=colors[i], alpha=0.8)
        
        # Highlight best
        if highlight_best:
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val == best_values[metric]:
                    bar.set_edgecolor('gold')
                    bar.set_linewidth(2)
    
    # Labels
    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


def plot_metric_heatmap(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Method Comparison Heatmap",
    figsize: Tuple[int, int] = None,
    cmap: str = "RdYlGn",
    annotate: bool = True,
    higher_better: Dict[str, bool] = None,
) -> plt.Figure:
    """
    Plot heatmap comparing methods across metrics.
    绘制热力图对比各方法的指标。
    
    Args:
        results_dict: Dict of {method_name: {metric_name: value}}
        metrics: List of metric names
        title: Plot title
        figsize: Figure size
        cmap: Colormap name
        annotate: Show values in cells
        higher_better: Dict of {metric_name: True/False}
    
    Returns:
        matplotlib Figure
    """
    methods = list(results_dict.keys())
    if metrics is None:
        metrics = list(list(results_dict.values())[0].keys())
    
    n_methods = len(methods)
    n_metrics = len(metrics)
    
    if figsize is None:
        figsize = (max(6, n_metrics * 1.2), max(4, n_methods * 0.5))
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Default higher_better
    if higher_better is None:
        higher_better = {m: not any(x in m.lower() for x in ['loss', 'error', 'fpr', 'nll', 'ece']) 
                        for m in metrics}
    
    # Build data matrix
    data = np.zeros((n_methods, n_metrics))
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            data[i, j] = results_dict[method].get(metric, 0)
    
    # Normalize each column (metric) to [0, 1] for visualization
    data_normalized = np.zeros_like(data)
    for j, metric in enumerate(metrics):
        col = data[:, j]
        if col.max() > col.min():
            normalized = (col - col.min()) / (col.max() - col.min())
            if not higher_better.get(metric, True):
                normalized = 1 - normalized  # Flip so "good" is always high
            data_normalized[:, j] = normalized
        else:
            data_normalized[:, j] = 0.5
    
    # Plot heatmap
    im = ax.imshow(data_normalized, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Annotate
    if annotate:
        for i in range(n_methods):
            for j in range(n_metrics):
                val = data[i, j]
                text_color = 'white' if data_normalized[i, j] < 0.3 or data_normalized[i, j] > 0.7 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', 
                       color=text_color, fontsize=8)
    
    # Labels
    ax.set_xticks(np.arange(n_metrics))
    ax.set_yticks(np.arange(n_methods))
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_title(title, fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative Performance (higher = better)', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_radar_chart(
    results_dict: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    title: str = "Method Comparison (Radar)",
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot radar chart comparing methods.
    绘制雷达图对比各方法。
    
    Args:
        results_dict: Dict of {method_name: {metric_name: value}}
        metrics: List of metric names
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    methods = list(results_dict.keys())
    if metrics is None:
        metrics = list(list(results_dict.values())[0].keys())
    
    n_metrics = len(metrics)
    
    # Compute angles
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Normalize data
    data_matrix = np.array([[results_dict[m].get(metric, 0) for metric in metrics] for m in methods])
    data_normalized = (data_matrix - data_matrix.min(axis=0)) / (data_matrix.max(axis=0) - data_matrix.min(axis=0) + 1e-8)
    
    for i, method in enumerate(methods):
        values = data_normalized[i].tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=method)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title(title, fontsize=12, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Demo / 演示
    results = {
        "MaxSoftmax": {"Accuracy": 0.95, "ECE": 0.08, "AUROC": 0.85, "FPR95": 0.35},
        "MC Dropout": {"Accuracy": 0.96, "ECE": 0.05, "AUROC": 0.89, "FPR95": 0.28},
        "EDL": {"Accuracy": 0.97, "ECE": 0.03, "AUROC": 0.92, "FPR95": 0.20},
        "Deep Ensembles": {"Accuracy": 0.98, "ECE": 0.02, "AUROC": 0.94, "FPR95": 0.15},
    }
    
    # Bar chart
    fig1 = plot_metric_comparison(results)
    plt.savefig("comparison_bar_demo.png", dpi=150, bbox_inches='tight')
    print("Saved: comparison_bar_demo.png")
    
    # Heatmap
    fig2 = plot_metric_heatmap(results)
    plt.savefig("comparison_heatmap_demo.png", dpi=150, bbox_inches='tight')
    print("Saved: comparison_heatmap_demo.png")
    
    plt.show()
