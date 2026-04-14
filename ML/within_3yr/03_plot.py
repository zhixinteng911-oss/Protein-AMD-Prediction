#!/usr/bin/env python3
"""Render the within_3yr SFS figure using the repo plotting style."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shared import WINDOW_CONFIG, get_output_dir

WINDOW = "within_3yr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the SFS results for the within-3-year model.")
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--display-top-n", type=int, default=None, help="Optionally truncate the displayed proteins to the first N ranked steps.")
    parser.add_argument("--output-stem", default="s05_fsf_plot", help="Base filename for the exported plot without extension.")
    args = parser.parse_args()

    out_dir = get_output_dir(args.base_out, WINDOW)
    rank_df = pd.read_csv(out_dir / "s03_reranked.csv")
    sfs_df = pd.read_csv(out_dir / "s04_fsf_results.csv")
    selected_df = pd.read_csv(out_dir / "02_selected_features.csv")
    optimal_k = len(selected_df)

    plot_df = sfs_df.merge(rank_df[["Features", "Cover"]], on="Features", how="left")
    plot_df["f_idx"] = range(1, len(plot_df) + 1)
    plot_df["AUC_lower"] = plot_df["AUC_mean"] - 1.96 * plot_df["AUC_std"]
    plot_df["AUC_upper"] = plot_df["AUC_mean"] + 1.96 * plot_df["AUC_std"]
    display_n = optimal_k if args.display_top_n is None else max(1, min(int(args.display_top_n), len(plot_df)))
    plot_df = plot_df.iloc[:display_n].copy()

    fig, ax = plt.subplots(figsize=(18, 7))
    palette = sns.color_palette("Blues", n_colors=len(plot_df))
    palette.reverse()
    sns.barplot(
        ax=ax,
        x="Features",
        y="Cover",
        hue="Features",
        dodge=False,
        legend=False,
        palette=palette,
        data=plot_df,
    )
    ax.set_ylim([0, plot_df["Cover"].max() * 1.15])
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["Features"], rotation=30, fontsize=12, ha="right")
    shown_optimal_k = min(optimal_k, len(plot_df))
    tick_colors = ["r"] * shown_optimal_k + ["k"] * (len(plot_df) - shown_optimal_k)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.set_ylabel("Predictor Importance", weight="bold", fontsize=18)
    ax.set_xlabel("")
    ax.grid(which="minor", alpha=0.2, linestyle=":")
    ax.grid(which="major", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(np.arange(shown_optimal_k), plot_df["AUC_mean"][:shown_optimal_k], "red", alpha=0.8, marker="o")
    if shown_optimal_k < len(plot_df):
        ax2.plot(np.arange(shown_optimal_k, len(plot_df)), plot_df["AUC_mean"][shown_optimal_k:], "black", alpha=0.8, marker="o")
        ax2.plot([shown_optimal_k - 1, shown_optimal_k], plot_df["AUC_mean"][shown_optimal_k - 1 : shown_optimal_k + 1], "black", alpha=0.8, marker="o")
    plt.fill_between(plot_df["f_idx"] - 1, plot_df["AUC_lower"], plot_df["AUC_upper"], color="tomato", alpha=0.2)
    ax2.set_ylabel("Cumulative AUC", weight="bold", fontsize=18)
    ax2.tick_params(axis="y", labelsize=14)
    fig.tight_layout()
    plt.xlim([-.6, len(plot_df) - .2])
    title = f"{WINDOW_CONFIG[WINDOW]['title']}: Forward Selection (Optimal K={optimal_k})"
    if display_n < optimal_k:
        title = f"{WINDOW_CONFIG[WINDOW]['title']}: Forward Selection (Top {display_n} Shown; Optimal K={optimal_k})"
    plt.title(title, fontweight="bold")
    plt.savefig(out_dir / f"{args.output_stem}.png", dpi=300)
    plt.savefig(out_dir / f"{args.output_stem}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
