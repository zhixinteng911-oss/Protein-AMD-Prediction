#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.metrics import roc_auc_score

from shared import LGBM_PROTEIN_PARAMS, fit_model, get_output_dir, make_folds, prepare_payload, predict_model

WINDOW = "within_3yr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-rank proteins for the within-3-year model.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    args = parser.parse_args()

    payload = prepare_payload(args.input_file, WINDOW)
    out_dir = get_output_dir(args.base_out, WINDOW)
    x_protein = payload["X_protein"]
    y = payload["y"]
    folds = make_folds(y, strict=True, label="feature re-ranking")
    params = {**LGBM_PROTEIN_PARAMS, "device": args.device}

    rank_res = []
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        x_train = x_protein.iloc[train_idx]
        x_val = x_protein.iloc[val_idx]
        y_train = y.iloc[train_idx].to_numpy(dtype=int)
        y_val = y.iloc[val_idx].to_numpy(dtype=int)

        model, means = fit_model(x_train, y_train, params)
        rank_res.append(
            pd.Series(
                model.booster_.feature_importance(importance_type="gain"),
                index=x_protein.columns,
            )
        )
        y_pred = predict_model(model, x_val, means)
        print(f"Fold {fold}/{len(folds)} - AUC: {roc_auc_score(y_val, y_pred):.4f}")

    avg_gain = pd.concat(rank_res, axis=1).mean(axis=1).sort_values(ascending=False)
    rank_df = avg_gain.reset_index()
    rank_df.columns = ["Features", "Gain"]
    rank_df["Cover"] = rank_df["Gain"] / rank_df["Gain"].max()
    rank_df.to_csv(out_dir / "s03_reranked.csv", index=False)


if __name__ == "__main__":
    main()
