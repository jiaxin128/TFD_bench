"""
Reliability Diagram / 可靠性图

Visualize model calibration by comparing predicted confidence with actual accuracy.
通过对比预测置信度和实际准确率来可视化模型校准。

Usage / 使用方法:
    from scripts.visualization import reliability
    fig = reliability.plot_reliability_diagram(confidences, accuracies)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import matplotlib


def compute_calibration_bins(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute calibration statistics for bins.
    计算分箱的校准统计量。
    
    Args:
        confidences: Predicted confidence scores [0, 1] / 预测置信度
        correctness: Binary correctness (1=correct, 0=wrong) / 是否正确
        n_bins: Number of bins / 分箱数
    
    Returns:
        bin_centers: Center of each bin / 每个分箱的中心
        bin_accuracies: Accuracy in each bin / 每个分箱的准确率
        bin_counts: Number of samples in each bin / 每个分箱的样本数
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
        if i == n_bins - 1:  # Include right boundary for last bin
            in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        
        bin_counts[i] = np.sum(in_bin)
        if bin_counts[i] > 0:
            bin_accuracies[i] = np.mean(correctness[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
    
    return bin_centers, bin_accuracies, bin_counts


def plot_reliability_diagram(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    figsize: Tuple[int, int] = (6, 5),
    color: str = "#3498db",
    show_gap: bool = True,
    show_counts: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot reliability diagram (calibration plot).
    绘制可靠性图（校准图）。
    
    Args:
        confidences: Predicted confidence scores [0, 1]
        correctness: Binary correctness (1=correct, 0=wrong)
        n_bins: Number of bins
        title: Plot title
        figsize: Figure size
        color: Bar color
        show_gap: Show calibration gap
        show_counts: Show sample counts
        ax: Existing axes (optional)
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    # Compute bins
    bin_centers, bin_accuracies, bin_counts = compute_calibration_bins(
        confidences, correctness, n_bins
    )
    
    bin_width = 1.0 / n_bins
    
    # Plot bars
    bars = ax.bar(
        bin_centers, bin_accuracies, 
        width=bin_width * 0.8, 
        color=color, 
        edgecolor='white',
        alpha=0.8,
        label='Accuracy'
    )
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    
    # Show calibration gap
    if show_gap:
        for i, (center, acc) in enumerate(zip(bin_centers, bin_accuracies)):
            if bin_counts[i] > 0:
                gap_color = '#e74c3c' if acc < center else '#27ae60'
                ax.plot([center, center], [center, acc], 
                       color=gap_color, linewidth=2, alpha=0.7)
    
    # Show sample counts
    if show_counts:
        for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
            if count > 0:
                ax.text(center, 0.02, f'{int(count)}', 
                       ha='center', va='bottom', fontsize=7, color='gray')
    
    # Compute ECE
    total_samples = np.sum(bin_counts)
    ece = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            ece += bin_counts[i] / total_samples * np.abs(bin_accuracies[i] - bin_centers[i])
    
    # Labels and title
    ax.set_xlabel('Confidence', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(f'{title}\nECE = {ece:.4f}', fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_multi_reliability(
    results_dict: dict,
    n_bins: int = 10,
    figsize: Tuple[int, int] = None,
    ncols: int = 3,
    title: str = "Reliability Diagrams Comparison"
) -> plt.Figure:
    """
    Plot multiple reliability diagrams for comparison.
    绘制多个可靠性图进行对比。
    
    Args:
        results_dict: Dict of {method_name: (confidences, correctness)}
        n_bins: Number of bins
        figsize: Figure size (auto if None)
        ncols: Number of columns
        title: Overall title
    
    Returns:
        matplotlib Figure
    """
    n_methods = len(results_dict)
    nrows = (n_methods + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_methods > 1 else [axes]
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (method_name, (confidences, correctness)) in enumerate(results_dict.items()):
        if i < len(axes):
            plot_reliability_diagram(
                confidences, correctness,
                n_bins=n_bins,
                title=method_name,
                color=colors[i % 10],
                ax=axes[i]
            )
    
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Demo / 演示
    np.random.seed(42)
    
    # Generate sample data / 生成示例数据
    n_samples = 1000
    confidences = np.random.beta(5, 2, n_samples)  # Overconfident model
    true_probs = confidences * 0.9  # Actual accuracy is lower
    correctness = np.random.binomial(1, true_probs)
    
    # Plot single diagram
    fig = plot_reliability_diagram(
        confidences, correctness,
        title="Sample Model"
    )
    plt.savefig("reliability_diagram_demo.png", dpi=150, bbox_inches='tight')
    print("Saved: reliability_diagram_demo.png")
    plt.show()
