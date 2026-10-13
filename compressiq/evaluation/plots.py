"""Research-grade publication plots for CompressIQ.

Generates:
1. Accuracy versus Model Size (MB) with Pareto frontier and budget markers
2. Accuracy versus Latency (ms)
3. Compression Ratio versus Accuracy Drop (%)
4. Layer-wise Sensitivity and Fragility Bar Chart
5. Ablation Comparison Plot across Baselines
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Clean research style constants
PALETTE = {
    "FP32 Baseline": "#1f2937",
    "Uniform INT8": "#2563eb",
    "Uniform Pruning": "#d97706",
    "Uniform Hybrid": "#7c3aed",
    "Sensitivity-Guided (Proposed)": "#059669",
    "Native INT8 PTQ": "#0284c7"
}


def setup_matplotlib_style():
    """Apply clean research typography and styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#d1d5db",
        "axes.linewidth": 1.0,
        "grid.color": "#f3f4f6",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "text.color": "#111827",
        "axes.labelcolor": "#374151",
        "xtick.color": "#4b5563",
        "ytick.color": "#4b5563",
    })


def plot_accuracy_vs_size(
    df: pd.DataFrame,
    target_budget_mb: Optional[float] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot Top-1 Accuracy (%) vs Model Size (MB)."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for method, row in df.iterrows():
        name = row.get("Method", method)
        size = row.get("model_size_mb", row.get("Size (MB)", 0.0))
        acc = row.get("top1_acc", row.get("Top-1 Acc (%)", 0.0))
        color = PALETTE.get(name, "#4b5563")

        marker = "*" if "Sensitivity" in name else "o"
        size_pt = 160 if "Sensitivity" in name else 100

        ax.scatter(size, acc, color=color, s=size_pt, label=name, zorder=4, edgecolors="none")
        ax.annotate(
            f" {name}\n ({size:.2f}MB, {acc:.1f}%)",
            (size, acc),
            fontsize=8,
            color="#374151",
            xytext=(6, -6),
            textcoords="offset points"
        )

    if target_budget_mb is not None:
        ax.axvline(
            x=target_budget_mb,
            color="#dc2626",
            linestyle="--",
            linewidth=1.5,
            label=f"Size Budget ({target_budget_mb:.1f} MB)",
            zorder=3
        )

    ax.set_xlabel("Model Size (MB)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Accuracy vs Model Size under Compression", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb", fontsize=8.5, loc="lower right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_accuracy_vs_latency(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot Top-1 Accuracy (%) vs Median Inference Latency (ms)."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for method, row in df.iterrows():
        name = row.get("Method", method)
        lat = row.get("median_latency_ms", row.get("Latency (ms)", 0.0))
        acc = row.get("top1_acc", row.get("Top-1 Acc (%)", 0.0))
        color = PALETTE.get(name, "#4b5563")

        ax.scatter(lat, acc, color=color, s=120, label=name, zorder=4)
        ax.annotate(
            f" {name}",
            (lat, acc),
            fontsize=8.5,
            color="#374151",
            xytext=(6, 4),
            textcoords="offset points"
        )

    ax.set_xlabel("Median Inference Latency (ms)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11, fontweight="bold")
    ax.set_title("Accuracy vs Latency Trade-off", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb", fontsize=8.5, loc="lower left")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_compression_vs_acc_drop(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot Compression Ratio (x) vs Accuracy Drop (percentage points)."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    for method, row in df.iterrows():
        name = row.get("Method", method)
        ratio = row.get("compression_ratio", row.get("Ratio", 1.0))
        drop = row.get("accuracy_drop", row.get("Acc Drop (%)", 0.0))
        color = PALETTE.get(name, "#4b5563")

        ax.scatter(ratio, drop, color=color, s=130, label=name, zorder=4)
        ax.annotate(
            f" {name}\n ({ratio:.1f}x, -{drop:.1f}%)",
            (ratio, drop),
            fontsize=8.5,
            color="#374151",
            xytext=(6, -6),
            textcoords="offset points"
        )

    ax.set_xlabel("Compression Ratio (Relative to FP32)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy Degradation (% Drop)", fontsize=11, fontweight="bold")
    ax.set_title("Compression Ratio vs Accuracy Degradation", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e5e7eb", fontsize=8.5, loc="upper left")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_layer_sensitivity_bars(
    profile_dict: Dict[str, Any],
    top_k: int = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot top-k most sensitive and most tolerant layers."""
    setup_matplotlib_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    records = profile_dict.get("records", [])
    if not records:
        return fig

    # Group records by layer: get maximum sensitivity score per layer
    layer_scores = {}
    for r in records:
        layer = r["layer_name"]
        score = r["sensitivity_score"]
        if layer not in layer_scores or score > layer_scores[layer]:
            layer_scores[layer] = score

    # Sort descending
    sorted_items = sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    layers = [item[0].replace("features.", "feat.") for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    # Invert for horizontal bar chart
    layers = layers[::-1]
    scores = scores[::-1]

    colors = ["#ef4444" if s > np.mean(scores) else "#10b981" for s in scores]
    bars = ax.barh(range(len(layers)), scores, color=colors, height=0.65, zorder=3)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=9)
    ax.set_xlabel("Sensitivity Score (ΔLoss / MB Saved)", fontsize=11, fontweight="bold")
    ax.set_title("Layer Sensitivity Profile (Fragile in Red, Tolerant in Green)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, axis="x")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + max(scores) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            fontsize=8,
            color="#374151"
        )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_ablation_summary(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Bar chart comparing accuracy and size across baselines."""
    setup_matplotlib_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=300)

    methods = df["Method"].tolist() if "Method" in df.columns else df.index.tolist()
    accs = df["top1_acc"].tolist() if "top1_acc" in df.columns else df["Top-1 Acc (%)"].tolist()
    sizes = df["model_size_mb"].tolist() if "model_size_mb" in df.columns else df["Size (MB)"].tolist()

    x = np.arange(len(methods))
    colors = [PALETTE.get(m, "#6b7280") for m in methods]

    # Left: Accuracy
    ax1.bar(x, accs, color=colors, width=0.55, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=25, ha="right", fontsize=8.5)
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Accuracy Comparison", fontsize=12, fontweight="bold")
    ax1.grid(True, axis="y")
    for i, v in enumerate(accs):
        ax1.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")

    # Right: Model Size
    ax2.bar(x, sizes, color=colors, width=0.55, zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=25, ha="right", fontsize=8.5)
    ax2.set_ylabel("Model Size (MB)", fontsize=10, fontweight="bold")
    ax2.set_title("Model Footprint (MB)", fontsize=12, fontweight="bold")
    ax2.grid(True, axis="y")
    for i, v in enumerate(sizes):
        ax2.text(i, v + 0.1, f"{v:.2f}M", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig
