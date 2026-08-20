"""
ROC Curve Comparison / ROC曲线对比

Compare OOD detection performance across methods using ROC curves.
使用ROC曲线对比各方法的OOD检测性能。

Usage / 使用方法:
    from scripts.visualization import roc
    fig = roc.plot_roc_curves(results_dict)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def plot_roc_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves for OOD Detection",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple methods.
    绘制多个方法的ROC曲线。
    
    Args:
        results_dict: Dict of {method_name: (labels, scores)}
            labels: Binary labels (0=ID, 1=OOD)
            scores: OOD scores (higher = more OOD)
        title: Plot title
        figsize: Figure size
        ax: Existing axes (optional)
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (method_name, (labels, scores)) in enumerate(results_dict.items()):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i % 10], linewidth=2,
                label=f'{method_name} (AUC = {roc_auc:.4f})')
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.5)')
    
    # Labels
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_pr_curves(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "Precision-Recall Curves for OOD Detection",
    figsize: Tuple[int, int] = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple methods.
    绘制多个方法的精确率-召回率曲线。
    
    Args:
        results_dict: Dict of {method_name: (labels, scores)}
        title: Plot title
        figsize: Figure size
        ax: Existing axes (optional)
    
    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i, (method_name, (labels, scores)) in enumerate(results_dict.items()):
        precision, recall, _ = precision_recall_curve(labels, scores)
        ap = average_precision_score(labels, scores)
        
        ax.plot(recall, precision, color=colors[i % 10], linewidth=2,
                label=f'{method_name} (AP = {ap:.4f})')
    
    # Labels
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig


def plot_roc_and_pr(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str = "OOD Detection Performance",
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot both ROC and PR curves side by side.
    并排绘制ROC和PR曲线。
    
    Args:
        results_dict: Dict of {method_name: (labels, scores)}
        title: Overall title
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    plot_roc_curves(results_dict, title="ROC Curves", ax=ax1)
    plot_pr_curves(results_dict, title="Precision-Recall Curves", ax=ax2)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def compute_fpr_at_tpr(labels: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    Compute FPR at a given TPR threshold.
    计算给定TPR阈值下的FPR。
    
    Args:
        labels: Binary labels
        scores: OOD scores
        target_tpr: Target TPR (default 0.95)
    
    Returns:
        FPR at target TPR
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.argmin(np.abs(tpr - target_tpr))
    return fpr[idx]


if __name__ == "__main__":
    # Demo / 演示
    np.random.seed(42)
    
    # Generate sample data / 生成示例数据
    n_id = 500
    n_ood = 500
    
    # Method 1: Good separation
    id_scores_1 = np.random.beta(2, 5, n_id)
    ood_scores_1 = np.random.beta(5, 2, n_ood)
    labels_1 = np.concatenate([np.zeros(n_id), np.ones(n_ood)])
    scores_1 = np.concatenate([id_scores_1, ood_scores_1])
    
    # Method 2: Medium separation
    id_scores_2 = np.random.beta(3, 4, n_id)
    ood_scores_2 = np.random.beta(4, 3, n_ood)
    labels_2 = labels_1.copy()
    scores_2 = np.concatenate([id_scores_2, ood_scores_2])
    
    results_dict = {
        "Method A (Good)": (labels_1, scores_1),
        "Method B (Medium)": (labels_2, scores_2),
    }
    
    # Plot
    fig = plot_roc_and_pr(results_dict)
    plt.savefig("roc_pr_curves_demo.png", dpi=150, bbox_inches='tight')
    print("Saved: roc_pr_curves_demo.png")
    plt.show()
