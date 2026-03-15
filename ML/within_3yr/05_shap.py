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
import shap
import warnings

from shared import LGBM_PROTEIN_PARAMS, WINDOW_CONFIG, fit_model, get_output_dir, make_folds, prepare_payload

WINDOW = "within_3yr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SHAP plots for the within-3-year model.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    args = parser.parse_args()

    payload = prepare_payload(args.input_file, WINDOW)
    out_dir = get_output_dir(args.base_out, WINDOW)
    y = payload["y"]
    x_all = payload["X_all"]
    selected_features = pd.read_csv(out_dir / "02_selected_features.csv")["feature"].tolist()
    folds = make_folds(y, strict=True, label="SHAP")
    params = {**LGBM_PROTEIN_PARAMS, "device": args.device}

    x_shap = x_all[selected_features].fillna(x_all[selected_features].mean())
    train_idx, test_idx = folds[-1]
    x_train = x_shap.iloc[train_idx]
    x_test = x_shap.iloc[test_idx]
    y_train = y.iloc[train_idx].to_numpy(dtype=int)

    model, _ = fit_model(x_train, y_train, params)
    explainer = shap.Explainer(model)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        shap_values = explainer(x_test)

    plt.figure(figsize=(10, 8))
    plt.title(f"{WINDOW_CONFIG[WINDOW]['title']}: SHAP Summary (Top {len(selected_features)})", fontsize=14, pad=20)
    shap.plots.beeswarm(shap_values, max_display=len(selected_features), order=np.arange(len(selected_features)), show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "s07_shap.png", dpi=300)
    plt.savefig(out_dir / "s07_shap.pdf")
    plt.close()


if __name__ == "__main__":
    main()
