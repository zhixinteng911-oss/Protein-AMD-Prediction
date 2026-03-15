#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from shared import MAX_FEATS, LGBM_PROTEIN_PARAMS, find_optimal_k, fit_model, get_output_dir, make_folds, prepare_payload, predict_model

WINDOW = "9plus_yr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forward selection for the beyond-9-year model.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--max-feats", type=int, default=MAX_FEATS)
    args = parser.parse_args()

    payload = prepare_payload(args.input_file, WINDOW)
    out_dir = get_output_dir(args.base_out, WINDOW)
    ranking = pd.read_csv(out_dir / "s03_reranked.csv")
    ranked_features = ranking["Features"].tolist()
    x_protein = payload["X_protein"]
    y = payload["y"]
    folds = make_folds(y, strict=True, label="feature selection")
    params = {**LGBM_PROTEIN_PARAMS, "device": args.device}

    fs_stats = []
    current_features = []
    for i, feature in enumerate(ranked_features[: args.max_feats], start=1):
        current_features.append(feature)
        fold_aucs = []
        for train_idx, val_idx in folds:
            x_train = x_protein[current_features].iloc[train_idx]
            x_val = x_protein[current_features].iloc[val_idx]
            y_train = y.iloc[train_idx].to_numpy(dtype=int)
            y_val = y.iloc[val_idx].to_numpy(dtype=int)
            model, means = fit_model(x_train, y_train, params)
            y_pred = predict_model(model, x_val, means)
            fold_aucs.append(roc_auc_score(y_val, y_pred))
        fs_stats.append(
            {
                "Features": feature,
                "AUC_mean": float(np.mean(fold_aucs)),
                "AUC_std": float(np.std(fold_aucs)),
                "n_features": i,
                **{f"AUC{j}": fold_aucs[j] for j in range(len(fold_aucs))},
            }
        )

    sfs_df = pd.DataFrame(fs_stats)
    optimal_k = find_optimal_k(sfs_df)
    selected_features = sfs_df["Features"].iloc[:optimal_k].tolist()

    sfs_df.to_csv(out_dir / "s04_fsf_results.csv", index=False)
    pd.DataFrame({"feature": selected_features}).to_csv(out_dir / "02_selected_features.csv", index=False)
    print(f"Optimal K={optimal_k}, Best AUC={sfs_df['AUC_mean'].max():.4f}")


if __name__ == "__main__":
    main()
