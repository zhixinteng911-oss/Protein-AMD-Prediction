#!/usr/bin/env python3
"""Render 5-fold OOF ROC curves for the within_3yr pipeline."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from shared import (
    LGBM_COMBINED_PARAMS,
    LGBM_DEMO_PARAMS,
    LGBM_PROTEIN_PARAMS,
    get_output_dir,
    get_roc_plot_style,
    make_folds,
    prepare_prediction_payload,
)

WINDOW = "within_3yr"
def fit_and_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    fill_values: pd.Series,
    params: dict,
) -> np.ndarray:
    model = LGBMClassifier(**params)
    model.fit(x_train.fillna(fill_values), y_train.to_numpy(dtype=int))
    return model.predict_proba(x_val.fillna(fill_values))[:, 1]


def build_cross_terms(
    x_protein: pd.DataFrame,
    x_demo: pd.DataFrame,
    cross_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    x_all = pd.concat([x_protein, x_demo], axis=1).reset_index(drop=True)
    cross_cols: list[str] = []
    if "Age_at_recruitment" in x_all.columns:
        for feature in cross_features:
            cross_col = f"{feature}_x_age"
            x_all[cross_col] = x_all[feature] * x_all["Age_at_recruitment"]
            cross_cols.append(cross_col)
    return x_all, cross_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 5-fold OOF ROC curves for the within-3-year model.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    args = parser.parse_args()

    payload = prepare_prediction_payload(args.input_file, WINDOW)
    out_dir = get_output_dir(args.base_out, WINDOW)
    selected_features = pd.read_csv(out_dir / "02_selected_features.csv")["feature"].tolist()

    y = payload["y"]
    x_protein = payload["X_protein"][selected_features].copy()
    x_demo = payload["X_demo"].copy()
    demo_cols = list(x_demo.columns)
    folds = make_folds(y, strict=True, label="ROC evaluation")
    cross_features = selected_features[:5]
    x_all, cross_cols = build_cross_terms(x_protein, x_demo, cross_features)

    combined_cols = list(dict.fromkeys(selected_features + demo_cols + cross_cols))

    fold_store = []
    for train_idx, val_idx in folds:
        y_train = y.iloc[train_idx].astype(int)
        y_val = y.iloc[val_idx].astype(int)
        x_protein_train = x_protein.iloc[train_idx]
        x_demo_train = x_demo[demo_cols].iloc[train_idx]
        x_all_train = x_all[combined_cols].iloc[train_idx]
        fill_prot = x_protein_train.mean()
        fill_demo = x_demo_train.mean()
        fill_combined = x_all_train.mean()
        protein_pred = fit_and_predict(
            x_protein_train,
            y_train,
            x_protein.iloc[val_idx],
            fill_prot,
            {**LGBM_PROTEIN_PARAMS, "device": args.device},
        )
        demo_pred = fit_and_predict(
            x_demo_train,
            y_train,
            x_demo[demo_cols].iloc[val_idx],
            fill_demo,
            {**LGBM_DEMO_PARAMS, "device": args.device},
        )
        combined_pred = fit_and_predict(
            x_all_train,
            y_train,
            x_all[combined_cols].iloc[val_idx],
            fill_combined,
            {**LGBM_COMBINED_PARAMS, "device": args.device},
        )
        fold_store.append(
            {
                "y_val": y_val,
                "p_protein": protein_pred,
                "p_demo": demo_pred,
                "p_combined": combined_pred,
                "auc_protein": roc_auc_score(y_val, protein_pred),
                "auc_demo": roc_auc_score(y_val, demo_pred),
                "auc_combined": roc_auc_score(y_val, combined_pred),
            }
        )

    roc_specs = [
        ("Protein", "#00CED1", "p_protein", "auc_protein", len(selected_features)),
        ("Demographic", "#32CD32", "p_demo", "auc_demo", len(demo_cols)),
        ("Combined", "#DC143C", "p_combined", "auc_combined", len(combined_cols)),
    ]

    rows = []
    style = get_roc_plot_style()
    fig, ax = plt.subplots(figsize=style["figsize"])
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.grid(True, color="lightgray", linestyle="-", linewidth=0.8, alpha=0.8)

    for name, color, pred_key, auc_key, n_features in roc_specs:
        all_y_true = np.concatenate([item["y_val"].to_numpy(dtype=int) for item in fold_store]).astype(int)
        all_y_prob = np.concatenate([item[pred_key] for item in fold_store]).astype(float)
        fold_aucs = [item[auc_key] for item in fold_store]
        auc = float(np.mean(fold_aucs))
        ci_low = float(np.mean(fold_aucs) - 1.96 * np.std(fold_aucs))
        ci_high = float(np.mean(fold_aucs) + 1.96 * np.std(fold_aucs))
        fpr, tpr, _ = roc_curve(
            all_y_true,
            all_y_prob,
            pos_label=1,
            drop_intermediate=style["roc_drop_intermediate"],
        )
        if style["curve_mode"] == "plot":
            ax.plot(
                fpr,
                tpr,
                color=color,
                lw=style["line_width"],
                label=f"{name} ({auc:.3f} [{ci_low:.3f}-{ci_high:.3f}])",
            )
        else:
            ax.step(
                fpr,
                tpr,
                where="post",
                color=color,
                lw=style["line_width"],
                label=f"{name} ({auc:.3f} [{ci_low:.3f}-{ci_high:.3f}])",
            )
        rows.append(
            {
                "window": WINDOW,
                "model": name,
                "AUC_mean": round(auc, 6),
                "CI_low": round(ci_low, 6),
                "CI_high": round(ci_high, 6),
                "n_features": n_features,
            }
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=style["diag_width"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate", fontsize=style["xlabel_size"])
    ax.set_ylabel("True Positive Rate", fontsize=style["ylabel_size"])
    ax.tick_params(axis="both", labelsize=style["tick_size"])
    ax.text(
        0.01,
        0.98,
        "Incident AMD\nwithin 3 years",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=style["label_text_size"],
    )
    legend = ax.legend(
        loc="lower right",
        fontsize=style["legend_size"],
        frameon=True,
        fancybox=style["legend_fancybox"],
        framealpha=style["legend_alpha"],
    )
    legend.get_frame().set_facecolor(style["legend_facecolor"])
    legend.get_frame().set_edgecolor(style["legend_edgecolor"])
    plt.tight_layout()
    plt.savefig(out_dir / "s06_roc.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "s06_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    roc_df = pd.DataFrame(rows)
    roc_df.to_csv(out_dir / "s06_roc_results.csv", index=False)
    print(roc_df.to_string(index=False))


if __name__ == "__main__":
    main()
