#!/usr/bin/env python3

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

WINDOW = "9plus_yr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the SFS results for the beyond-9-year model.")
    parser.add_argument("--base-out", default="outputs")
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

    fig, ax = plt.subplots(figsize=(18, 7))
    palette = sns.color_palette("Blues", n_colors=len(plot_df))
    palette.reverse()
    sns.barplot(ax=ax, x="Features", y="Cover", palette=palette, data=plot_df)
    ax.set_ylim([0, plot_df["Cover"].max() * 1.15])
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xticklabels(plot_df["Features"], rotation=30, fontsize=12, ha="right")
    tick_colors = ["r"] * optimal_k + ["k"] * (len(plot_df) - optimal_k)
    for tick, color in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(color)
    ax.set_ylabel("Predictor Importance", weight="bold", fontsize=18)
    ax.set_xlabel("")
    ax.grid(which="minor", alpha=0.2, linestyle=":")
    ax.grid(which="major", alpha=0.5, linestyle="--")
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(np.arange(optimal_k), plot_df["AUC_mean"][:optimal_k], "red", alpha=0.8, marker="o")
    if optimal_k < len(plot_df):
        ax2.plot(np.arange(optimal_k, len(plot_df)), plot_df["AUC_mean"][optimal_k:], "black", alpha=0.8, marker="o")
        ax2.plot([optimal_k - 1, optimal_k], plot_df["AUC_mean"][optimal_k - 1 : optimal_k + 1], "black", alpha=0.8, marker="o")
    plt.fill_between(plot_df["f_idx"] - 1, plot_df["AUC_lower"], plot_df["AUC_upper"], color="tomato", alpha=0.2)
    ax2.set_ylabel("Cumulative AUC", weight="bold", fontsize=18)
    ax2.tick_params(axis="y", labelsize=14)
    fig.tight_layout()
    plt.xlim([-.6, len(plot_df) - .2])
    plt.title(f"{WINDOW_CONFIG[WINDOW]['title']}: Forward Selection (Optimal K={optimal_k})", fontweight="bold")
    plt.savefig(out_dir / "s05_fsf_plot.png", dpi=300)
    plt.savefig(out_dir / "s05_fsf_plot.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
