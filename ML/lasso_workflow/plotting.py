from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from .config import MODEL_LABELS, WINDOWS

AVAILABLE_FONTS = {font.name for font in font_manager.fontManager.ttflist}
PLOT_FONT = "Arial" if "Arial" in AVAILABLE_FONTS else "DejaVu Sans"
plt.rcParams.update({"font.family": PLOT_FONT, "font.size": 8, "pdf.fonttype": 42})


def _save(fig, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_auc_label(row: pd.Series) -> str:
    return (
        f"{row['model']} {row['auc']:.2f} "
        f"[{row['auc_ci_low']:.2f}-{row['auc_ci_high']:.2f}]"
    )


def plot_roc(
    predictions: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    out_dir: Path,
    *,
    panel_id: str = "selected_panel",
) -> None:
    colors = {
        "protein": "#2C7FB8",
        "demographic": "#6E6E6E",
        "combined": "#D95F5F",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharex=True, sharey=True)
    for ax, (window, label) in zip(axes, WINDOWS):
        for model_key in ("protein", "demographic", "combined"):
            pred_col = f"p_{model_key}"
            sub = predictions[
                predictions["panel_id"].eq(panel_id)
                & predictions["window"].eq(window)
                & predictions[pred_col].notna()
            ].copy()
            if sub.empty or sub["target_y"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(sub["target_y"].astype(int), sub[pred_col].astype(float))
            metric = pooled_metrics[
                pooled_metrics["panel_id"].eq(panel_id)
                & pooled_metrics["window"].eq(window)
                & pooled_metrics["model_key"].eq(model_key)
            ]
            if metric.empty:
                label_text = MODEL_LABELS[model_key]
            else:
                label_text = _format_auc_label(metric.iloc[0])
            ax.plot(fpr, tpr, color=colors[model_key], lw=1.8, label=label_text)
        ax.plot([0, 1], [0, 1], "--", color="0.65", lw=0.9)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("False positive rate", fontweight="bold")
        ax.grid(alpha=0.18, linewidth=0.6)
    axes[0].set_ylabel("True positive rate", fontweight="bold")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    axes[1].legend(frameon=False, loc="lower right", fontsize=8)
    axes[2].legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, f"fig4b_selected_panel_outer_heldout_roc")


def plot_metric_summary(pooled_metrics: pd.DataFrame, out_dir: Path) -> None:
    selected_panel = pooled_metrics[pooled_metrics["panel_id"].eq("selected_panel")].copy()
    if selected_panel.empty:
        return
    order = [window for window, _ in WINDOWS]
    labels = [label for _, label in WINDOWS]
    model_order = ["protein", "demographic", "combined"]
    offsets = {"protein": -0.22, "demographic": 0.0, "combined": 0.22}
    colors = {"protein": "#2C7FB8", "demographic": "#6E6E6E", "combined": "#D95F5F"}
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for model_key in model_order:
        sub = selected_panel[selected_panel["model_key"].eq(model_key)].set_index("window").reindex(order)
        x = np.arange(len(order)) + offsets[model_key]
        y = sub["auc"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                np.maximum(0.0, y - sub["auc_ci_low"].to_numpy(dtype=float)),
                np.maximum(0.0, sub["auc_ci_high"].to_numpy(dtype=float) - y),
            ]
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            color=colors[model_key],
            capsize=3,
            label=MODEL_LABELS[model_key],
        )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 0.9)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "supp_selected_panel_metric_summary")


def plot_consensus_coefficients(
    coefficients: pd.DataFrame,
    out_dir: Path,
    *,
    windows: list[str],
) -> None:
    if coefficients.empty:
        return
    window_labels = dict(WINDOWS)
    colors = {
        "exact_0_3": "#2C7FB8",
        "exact_3_9": "#D95F5F",
        "exact_gt9": "#6E6E6E",
    }
    offsets = np.linspace(-0.24, 0.24, len(windows))
    ordered = coefficients.sort_values("consensus_rank", ascending=False).reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(6.2, max(3.0, 0.32 * len(ordered))))
    for offset, window in zip(offsets, windows):
        column = f"mean_standardized_coefficient_{window}"
        ax.scatter(
            ordered[column],
            y + offset,
            s=24,
            color=colors.get(window, "#4C78A8"),
            label=window_labels.get(window, window),
            zorder=3,
        )
    ax.axvline(0, color="0.55", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["display_name"])
    ax.set_xlabel("Mean standardized LASSO coefficient")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_dir, "fig4c_consensus_coefficients")


def plot_feature_frequency(
    frequency: pd.DataFrame,
    out_dir: Path,
    *,
    panel_id: str = "selected_panel",
) -> None:
    sub = frequency[frequency["panel_id"].eq(panel_id)].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["outer_fold_selection_count", "feature"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(5.8, max(2.5, 0.26 * len(sub))))
    colors = ["#D95F5F" if feature == "nectin2" else "#4C78A8" for feature in sub["feature"]]
    ax.barh(sub["display_name"], sub["outer_fold_selection_frequency"], color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Outer-fold selection frequency")
    ax.set_title(f"{panel_id} panel retention")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    _save(fig, out_dir, f"supp_{panel_id}_retention")
