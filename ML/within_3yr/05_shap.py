#!/usr/bin/env python3
"""Render fold-aggregated SHAP summaries for the within_3yr pipeline."""

import argparse
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier

from shared import (
    PARTICIPANT_ID_COL,
    feature_display_name,
    LGBM_PROTEIN_PARAMS,
    get_output_dir,
    make_folds,
    prepare_prediction_payload,
)

WINDOW = "within_3yr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP plots for the within-3-year full-data 5-fold OOF model.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    args = parser.parse_args()

    payload = prepare_prediction_payload(args.input_file, WINDOW)
    out_dir = get_output_dir(args.base_out, WINDOW)
    selected_features = pd.read_csv(out_dir / "02_selected_features.csv")["feature"].tolist()

    y = payload["y"]
    window_df = payload["df"].reset_index(drop=True)
    x_protein = payload["X_protein"][selected_features].copy()
    folds = make_folds(y, strict=True, label="SHAP")
    params = {**LGBM_PROTEIN_PARAMS, "device": args.device}

    display_names = [feature_display_name(col) for col in selected_features]
    shap_blocks = []
    x_display_blocks = []
    x_raw_blocks = []
    id_blocks = []
    for train_idx, val_idx in folds:
        x_train = x_protein.iloc[train_idx].copy()
        x_val = x_protein.iloc[val_idx].copy()
        y_train = y.iloc[train_idx].to_numpy(dtype=int)
        train_means = x_train.mean()
        x_train_filled = x_train.fillna(train_means).fillna(0)
        x_val_filled = x_val.fillna(train_means).fillna(0)
        x_train_filled.columns = display_names
        x_val_filled.columns = display_names

        fitted_model = LGBMClassifier(**params)
        fitted_model.fit(x_train_filled, y_train)

        explainer = shap.Explainer(fitted_model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            shap_values = explainer(x_val_filled)
        shap_blocks.append(np.asarray(shap_values.values))
        x_display_blocks.append(x_val_filled)
        x_raw_blocks.append(x_val_filled.rename(columns=dict(zip(display_names, selected_features))))
        if PARTICIPANT_ID_COL in window_df.columns:
            id_blocks.append(window_df.iloc[val_idx][PARTICIPANT_ID_COL].reset_index(drop=True))
        else:
            id_blocks.append(pd.Series(window_df.index[val_idx], name="X"))

    all_shap_values = np.vstack(shap_blocks)
    x_test_display = pd.concat(x_display_blocks, axis=0).reset_index(drop=True)
    x_test_raw = pd.concat(x_raw_blocks, axis=0).reset_index(drop=True)
    sample_ids = pd.concat(id_blocks, axis=0).reset_index(drop=True)
    mean_abs = np.abs(all_shap_values).mean(axis=0)
    shap_df = pd.DataFrame(
        {
            "feature": display_names,
            "mean_abs_shap": mean_abs,
            "panel_order": np.arange(1, len(display_names) + 1),
        }
    )
    shap_df.to_csv(out_dir / "s07_shap_top_features.csv", index=False)

    shap_detail = pd.DataFrame({"X": sample_ids.astype(str)})
    for feature in selected_features:
        shap_detail[feature] = x_test_raw[feature].to_numpy()
    for idx, feature in enumerate(selected_features):
        shap_detail[f"{feature}_SHAP"] = all_shap_values[:, idx]
    shap_detail.to_csv(out_dir / "s07_shap_detail.csv", index=False)

    plt.figure(figsize=(7.2, 8.4))
    shap.summary_plot(
        all_shap_values,
        x_test_display,
        feature_names=display_names,
        max_display=len(selected_features),
        sort=False,
        show=False,
    )
    ax = plt.gca()
    ax.set_title("SHAP Feature Importance (3-Year)", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "s07_shap.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "s07_shap.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
