#!/usr/bin/env python3
"""Run 5-fold sequential forward selection for within_3yr."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from shared import (
    LGBM_PROTEIN_PARAMS,
    find_optimal_k,
    fit_model,
    get_output_dir,
    make_folds,
    prepare_prediction_payload,
    predict_model,
)

WINDOW = "within_3yr"


def write_sfs_progress(
    out_dir: Path,
    fs_stats: list[dict[str, float | str]],
    *,
    current_index: int,
    total_features: int,
) -> Path:
    """Persist an incremental checkpoint so long SFS runs stay observable."""
    progress_path = out_dir / "s04_fsf_results.partial.csv"
    pd.DataFrame(fs_stats).to_csv(progress_path, index=False)
    latest = fs_stats[-1]
    print(
        f"SFS progress {current_index}/{total_features}: "
        f"{latest['Features']} mean_auc={latest['AUC_mean']:.4f}",
        flush=True,
    )
    return progress_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forward selection for the within-3-year model.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    parser.add_argument("--max-feats", type=int, default=None)
    args = parser.parse_args()

    payload = prepare_prediction_payload(args.input_file, WINDOW)
    out_dir = get_output_dir(args.base_out, WINDOW)
    x_protein = payload["X_protein"]
    y = payload["y"]
    ranking = pd.read_csv(out_dir / "s03_reranked.csv")
    ranked_features = [feature for feature in ranking["Features"].tolist() if feature in x_protein.columns]
    if args.max_feats is not None:
        ranked_features = ranked_features[: args.max_feats]
    folds = make_folds(y, strict=True, label="feature selection")
    params = {**LGBM_PROTEIN_PARAMS, "device": args.device}

    fs_stats = []
    current_features = []
    for i, feature in enumerate(ranked_features, start=1):
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
        write_sfs_progress(out_dir, fs_stats, current_index=i, total_features=len(ranked_features))

    sfs_df = pd.DataFrame(fs_stats)
    optimal_k = find_optimal_k(sfs_df)
    selected_features = sfs_df["Features"].iloc[:optimal_k].tolist()

    sfs_df.to_csv(out_dir / "s04_fsf_results.csv", index=False)
    partial_path = out_dir / "s04_fsf_results.partial.csv"
    if partial_path.exists():
        partial_path.unlink()
    pd.DataFrame({"feature": selected_features}).to_csv(out_dir / "02_selected_features.csv", index=False)
    print(f"Optimal K={optimal_k}, Best AUC={sfs_df['AUC_mean'].max():.4f}")


if __name__ == "__main__":
    main()
