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
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from shared import (
    WINDOW_CONFIG,
    LGBM_COMBINED_PARAMS,
    LGBM_DEMO_PARAMS,
    LGBM_PROTEIN_PARAMS,
    get_output_dir,
    make_folds,
    prepare_payload,
)

WINDOW = "9plus_yr"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ROC curves for the beyond-9-year model.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    args = parser.parse_args()

    payload = prepare_payload(args.input_file, WINDOW)
    out_dir = get_output_dir(args.base_out, WINDOW)
    y = payload["y"]
    x_all = payload["X_all"]
    demo_cols = list(payload["demo_cols"])
    cross_cols = list(payload["cross_cols"])
    selected_features = pd.read_csv(out_dir / "02_selected_features.csv")["feature"].dropna().tolist()
    if not selected_features:
        raise ValueError("Run 02_sfs.py before 04_roc.py.")
    optimal_k = len(selected_features)

    fill_prot = x_all[selected_features].mean()
    fill_demo = x_all[demo_cols].mean()
    combined_cols = list(dict.fromkeys(selected_features + demo_cols + cross_cols))
    fill_combined = x_all[combined_cols].mean()

    fold_store = []
    folds = make_folds(y, strict=True, label="ROC evaluation")
    for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
        y_train = y.iloc[train_idx].astype(int)
        y_val = y.iloc[val_idx].astype(int)

        protein_pred = fit_and_predict(
            x_all[selected_features].iloc[train_idx],
            y_train,
            x_all[selected_features].iloc[val_idx],
            fill_prot,
            {**LGBM_PROTEIN_PARAMS, "device": args.device},
        )
        protein_auc = roc_auc_score(y_val, protein_pred)

        demo_pred = fit_and_predict(
            x_all[demo_cols].iloc[train_idx],
            y_train,
            x_all[demo_cols].iloc[val_idx],
            fill_demo,
            {**LGBM_DEMO_PARAMS, "device": args.device},
        )
        demo_auc = roc_auc_score(y_val, demo_pred)

        combined_pred = fit_and_predict(
            x_all[combined_cols].iloc[train_idx],
            y_train,
            x_all[combined_cols].iloc[val_idx],
            fill_combined,
            {**LGBM_COMBINED_PARAMS, "device": args.device},
        )
        combined_auc = roc_auc_score(y_val, combined_pred)

        fold_store.append(
            {
                "y_val": y_val,
                "p_protein": protein_pred,
                "p_demo": demo_pred,
                "p_combined": combined_pred,
                "auc_protein": protein_auc,
                "auc_demo": demo_auc,
                "auc_combined": combined_auc,
            }
        )

        print(
            f"Fold {fold_idx}: Protein={protein_auc:.4f}  "
            f"Demo={demo_auc:.4f}  "
            f"Combined={combined_auc:.4f}"
        )

    models = [
        {"name": f"Protein Only (Top {optimal_k})", "color": "#00CED1", "key": "p_protein", "auc_key": "auc_protein"},
        {"name": "Demographics (6 features)", "color": "#32CD32", "key": "p_demo", "auc_key": "auc_demo"},
        {"name": f"Combined (Demo+Top {optimal_k})", "color": "#DC143C", "key": "p_combined", "auc_key": "auc_combined"},
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xticks(np.arange(0, 1.2, 0.2))
    ax.set_yticks(np.arange(0, 1.2, 0.2))
    ax.grid(True, color="lightgray", linestyle="-", linewidth=0.8, alpha=0.8)

    roc_rows = []
    for model_cfg in models:
        all_y_true = np.concatenate([item["y_val"].to_numpy(dtype=int) for item in fold_store]).astype(int)
        all_y_prob = np.concatenate([item[model_cfg["key"]] for item in fold_store]).astype(float)
        aucs = [item[model_cfg["auc_key"]] for item in fold_store]

        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob, pos_label=1)
        mean_auc = np.round(np.mean(aucs), 3)
        ci_low = np.round(np.mean(aucs) - 1.96 * np.std(aucs), 3)
        ci_high = np.round(np.mean(aucs) + 1.96 * np.std(aucs), 3)
        ax.plot(fpr, tpr, color=model_cfg["color"], lw=2, label=f"{model_cfg['name']} ({mean_auc} [{ci_low}-{ci_high}])")

        roc_rows.append(
            {
                "window": WINDOW,
                "model": model_cfg["name"],
                "AUC_mean": mean_auc,
                "CI_low": ci_low,
                "CI_high": ci_high,
                "AUC_std": np.round(np.std(aucs), 3),
                "optimal_k": optimal_k,
                "n_features": len(selected_features)
                if "Protein" in model_cfg["name"] and "Combined" not in model_cfg["name"]
                else len(demo_cols)
                if "Demo" in model_cfg["name"] and "Combined" not in model_cfg["name"]
                else len(combined_cols),
                "combined_source": "",
            }
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random (0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=14)
    ax.set_title(f"ROC Curves: {WINDOW_CONFIG[WINDOW]['title']}", fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="lower right", fontsize=11, frameon=True, edgecolor="gray", fancybox=False)
    plt.tight_layout()
    plt.savefig(out_dir / "s06_roc.pdf", dpi=300, bbox_inches="tight", format="pdf")
    plt.savefig(out_dir / "s06_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    roc_df = pd.DataFrame(roc_rows)
    roc_df.to_csv(out_dir / "s06_roc_results.csv", index=False)
    print("\n[S06] Summary:")
    print(roc_df[["model", "AUC_mean", "CI_low", "CI_high", "combined_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
