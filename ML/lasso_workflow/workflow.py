from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .config import WINDOWS
from .data import read_analysis_frame, resolve_path
from .modeling import (
    SelectionConfig,
    compress_candidate_pool,
    feature_frequency,
    feature_display_name,
    fit_models_for_window,
    integrate_window_rankings,
    make_outer_folds,
    metric_values,
    run_window_stability_train_only,
    summarize_predictions,
)
from .plotting import (
    plot_consensus_coefficients,
    plot_feature_frequency,
    plot_metric_summary,
    plot_roc,
)


def validate_args(args) -> None:
    if int(args.n_subsamples) <= 0:
        raise ValueError("--n-subsamples must be positive.")
    if not 0 < float(args.sample_frac) <= 1:
        raise ValueError("--sample-frac must be in (0, 1].")
    if float(args.control_case_ratio) < 0:
        raise ValueError("--control-case-ratio must be non-negative.")
    if int(args.outer_folds) < 2:
        raise ValueError("--outer-folds must be at least 2.")
    if float(args.lasso_c) <= 0 or float(args.logistic_c) <= 0:
        raise ValueError("Regularization strengths must be positive.")
    if int(args.bootstrap) < 0:
        raise ValueError("--bootstrap must be non-negative.")
    if int(args.candidate_pool_size) < int(args.panel_size):
        raise ValueError("--candidate-pool-size must be at least --panel-size.")


def _write_config(
    args,
    *,
    out_dir: Path,
    input_path: Path,
    proteins_path: Path,
    df: pd.DataFrame,
    proteins: list[str],
    demographics: list[str],
    adjustment_covariates: list[str],
    resolved_columns: dict[str, str],
) -> None:
    config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "input_file": str(input_path),
        "protein_manifest": str(proteins_path),
        "out_dir": str(out_dir),
        "id_col": args.id_col,
        "windows": list(args.windows),
        "control_mode": args.control_mode,
        "candidate_pool_size": int(args.candidate_pool_size),
        "panel_size": int(args.panel_size),
        "outer_folds": int(args.outer_folds),
        "seed": int(args.seed),
        "n_subsamples": int(args.n_subsamples),
        "sample_frac": float(args.sample_frac),
        "control_case_ratio": float(args.control_case_ratio),
        "lasso_c": float(args.lasso_c),
        "lasso_max_iter": int(args.lasso_max_iter),
        "lasso_tol": float(args.lasso_tol),
        "logistic_c": float(args.logistic_c),
        "final_penalty": "l1",
        "class_weight": args.class_weight,
        "bootstrap": int(args.bootstrap),
        "rows": int(len(df)),
        "incident_cases_total": int(df["target_y"].sum()),
        "protein_columns": int(len(proteins)),
        "demographic_columns": demographics,
        "stability_adjustment_covariates": adjustment_covariates,
        "resolved_columns": resolved_columns,
        "method_note": (
            "Outer-training LASSO stability selection, top-20 candidate "
            "retention, secondary L1 compression, and held-out evaluation."
        ),
        "sklearn_version": sklearn.__version__,
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_result_readme(out_dir: Path, pooled: pd.DataFrame, freq_df: pd.DataFrame) -> None:
    selected_panel = pooled[pooled["panel_id"].eq("selected_panel")].copy()
    lines = [
        "# Machine-learning validation",
        "",
        "Within each outer training set, stability selection retained 20 candidates",
        "and a second L1 model produced a fold-specific 10-protein panel. Performance was",
        "calculated from pooled held-out predictions.",
        "",
        "## Pooled metrics",
        "",
    ]
    if not selected_panel.empty:
        lines.append(
            selected_panel[
                [
                    "window_label",
                    "model",
                    "n",
                    "cases",
                    "auc",
                    "auc_ci_low",
                    "auc_ci_high",
                    "average_precision",
                    "brier",
                ]
            ].to_string(index=False)
        )
    nect = freq_df[freq_df["is_nectin2"].eq(True)].copy() if not freq_df.empty else pd.DataFrame()
    if not nect.empty:
        lines.extend(["", "## NECTIN2 retention", ""])
        lines.append(
            nect[
                [
                    "panel_id",
                    "outer_fold_selection_count",
                    "outer_fold_selection_frequency",
                ]
            ].to_string(index=False)
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_outer_training_stability(
    rankings: pd.DataFrame,
    windows: list[str],
    n_subsamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rankings.groupby(["window", "feature", "display_name"], as_index=False)
        .agg(
            mean_stability_rank=("stability_rank", "mean"),
            mean_selection_frequency=("selection_frequency", "mean"),
            mean_abs_coef=("mean_abs_coef", "mean"),
            mean_signed_coef=("mean_signed_coef", "mean"),
            outer_folds_observed=("outer_fold", "nunique"),
        )
    )
    summary["mean_selection_count"] = (
        summary["mean_selection_frequency"] * int(n_subsamples)
    )
    integrated = (
        summary.groupby(["feature", "display_name"], as_index=False)
        .agg(
            integrated_lasso_score=("mean_abs_coef", "sum"),
            mean_selection_frequency=("mean_selection_frequency", "mean"),
            best_mean_stability_rank=("mean_stability_rank", "min"),
            windows_observed=("window", "nunique"),
        )
        .sort_values(
            [
                "integrated_lasso_score",
                "mean_selection_frequency",
                "best_mean_stability_rank",
                "feature",
            ],
            ascending=[False, False, True, True],
        )
        .reset_index(drop=True)
    )
    integrated.insert(0, "integrated_rank", np.arange(1, len(integrated) + 1))
    if set(summary["window"].unique()) != set(windows):
        raise RuntimeError("Outer-training stability summary is missing a temporal window.")
    return summary, integrated


def _summarize_consensus_coefficients(
    coefficients: pd.DataFrame,
    frequency: pd.DataFrame,
    *,
    windows: list[str],
    n_folds: int,
    panel_size: int,
) -> pd.DataFrame:
    if coefficients.empty or frequency.empty:
        return pd.DataFrame()
    count_map = frequency.set_index("feature")["outer_fold_selection_count"].to_dict()
    rows: list[dict[str, object]] = []
    for feature in frequency["feature"].astype(str):
        feature_rows = coefficients[coefficients["feature"].eq(feature)]
        record: dict[str, object] = {
            "feature": feature,
            "display_name": feature_display_name(feature),
            "outer_fold_selection_count": int(count_map.get(feature, 0)),
            "outer_fold_selection_frequency": float(count_map.get(feature, 0)) / int(n_folds),
        }
        integrated_abs = 0.0
        for window in windows:
            values = np.zeros(int(n_folds), dtype=float)
            window_rows = feature_rows[feature_rows["window"].eq(window)]
            for _, row in window_rows.iterrows():
                values[int(row["outer_fold"]) - 1] = float(row["standardized_coefficient"])
            record[f"mean_standardized_coefficient_{window}"] = float(values.mean())
            integrated_abs += float(np.abs(values).mean())
        record["integrated_mean_abs_coefficient"] = integrated_abs
        rows.append(record)
    consensus = pd.DataFrame(rows).sort_values(
        ["outer_fold_selection_count", "integrated_mean_abs_coefficient", "feature"],
        ascending=[False, False, True],
    ).head(int(panel_size)).reset_index(drop=True)
    consensus.insert(0, "consensus_rank", np.arange(1, len(consensus) + 1))
    return consensus


def run_workflow(args) -> dict[str, pd.DataFrame]:
    validate_args(args)
    script_dir = Path(__file__).resolve().parents[1]
    input_path = resolve_path(args.input, script_dir).expanduser().resolve()
    proteins_path = resolve_path(args.proteins, script_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser()
    if not out_dir.is_absolute():
        out_dir = Path.cwd() / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df, proteins, demographics, adjustment_covariates, id_col, resolved_columns = read_analysis_frame(
        input_path,
        proteins_file=proteins_path,
        id_col=args.id_col,
    )
    folds = make_outer_folds(df, int(args.outer_folds), int(args.seed))
    selection_config = SelectionConfig(
        n_subsamples=int(args.n_subsamples),
        sample_frac=float(args.sample_frac),
        control_case_ratio=float(args.control_case_ratio),
        lasso_c=float(args.lasso_c),
        lasso_max_iter=int(args.lasso_max_iter),
        lasso_tol=float(args.lasso_tol),
        class_weight=args.class_weight,
        seed=int(args.seed),
    )

    prediction_frames: list[pd.DataFrame] = []
    split_metric_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    qc_rows: list[dict[str, object]] = []
    ranking_blocks: list[pd.DataFrame] = []
    integrated_blocks: list[pd.DataFrame] = []
    candidate_pool_rows: list[dict[str, object]] = []
    compression_blocks: list[pd.DataFrame] = []
    coefficient_blocks: list[pd.DataFrame] = []

    for fold_id, (train_idx, test_idx) in enumerate(folds, start=1):
        print(f"Outer fold {fold_id}/{len(folds)}", flush=True)
        fold_out = out_dir / f"outer_fold_{fold_id}"
        fold_out.mkdir(parents=True, exist_ok=True)
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        rankings: dict[str, pd.DataFrame] = {}
        ranking_qc_rows: list[dict[str, object]] = []
        for window_index, window in enumerate(args.windows):
            ranking, ranking_qc = run_window_stability_train_only(
                train_df,
                proteins,
                adjustment_covariates,
                window,
                control_mode=args.control_mode,
                config=selection_config,
                seed=int(args.seed) + fold_id * 10_000 + window_index * 1_000,
            )
            rankings[window] = ranking
            ranking.to_csv(fold_out / f"{window}_train_stability_ranking.csv", index=False)
            ranking_block = ranking.copy()
            ranking_block.insert(0, "window", window)
            ranking_block.insert(0, "outer_fold", fold_id)
            ranking_blocks.append(ranking_block)
            ranking_qc["outer_fold"] = fold_id
            ranking_qc_rows.append(ranking_qc)
        pd.DataFrame(ranking_qc_rows).to_csv(
            fold_out / "window_train_stability_qc.csv",
            index=False,
        )
        integrated = integrate_window_rankings(rankings, list(args.windows))
        integrated.to_csv(fold_out / "integrated_lasso_stability_ranking.csv", index=False)
        integrated_block = integrated.copy()
        integrated_block.insert(0, "outer_fold", fold_id)
        integrated_blocks.append(integrated_block)

        candidate_pool = integrated["feature"].astype(str).head(int(args.candidate_pool_size)).tolist()
        selected, compression = compress_candidate_pool(
            train_df,
            candidate_pool,
            adjustment_covariates,
            list(args.windows),
            control_mode=args.control_mode,
            config=selection_config,
            panel_size=int(args.panel_size),
            seed=int(args.seed) + fold_id * 100_000,
        )
        panel_id = "selected_panel"
        candidate_pool_rows.append(
            {
                "outer_fold": fold_id,
                "candidate_pool_size": len(candidate_pool),
                "candidate_pool": ";".join(candidate_pool),
            }
        )
        compression.insert(0, "outer_fold", fold_id)
        compression_blocks.append(compression)
        compression.to_csv(fold_out / "candidate_pool_l1_compression.csv", index=False)
        panel_rows.append(
            {
                "outer_fold": fold_id,
                "panel_id": panel_id,
                "candidate_pool_size": len(candidate_pool),
                "panel_size": len(selected),
                "selected_features": ";".join(selected),
                "contains_nectin2": "nectin2" in selected,
                "nectin2_stability_rank": (
                    int(integrated.loc[integrated["feature"].eq("nectin2"), "integrated_rank"].iloc[0])
                    if integrated["feature"].eq("nectin2").any()
                    else np.nan
                ),
            }
        )
        for window_index, window in enumerate(args.windows):
            train_window_df, train_y, train_qc = build_window(train_df, window, args.control_mode)
            test_window_df, test_y, test_qc = build_window(test_df, window, args.control_mode)
            qc_rows.append(
                {
                    "outer_fold": fold_id,
                    "panel_id": panel_id,
                    "candidate_pool_size": len(candidate_pool),
                    "panel_size": len(selected),
                    "window": window,
                    "window_label": train_qc["window_label"],
                    "train_n": int(train_qc["n"]),
                    "train_cases": int(train_qc["cases"]),
                    "train_controls": int(train_qc["controls"]),
                    "test_n": int(test_qc["n"]),
                    "test_cases": int(test_qc["cases"]),
                    "test_controls": int(test_qc["controls"]),
                }
            )
            if train_y.nunique() < 2 or test_y.nunique() < 2:
                raise RuntimeError(
                    f"Outer fold {fold_id}, window {window} does not contain both outcome classes."
                )
            predictions, protein_coefficients = fit_models_for_window(
                train_window_df.reset_index(drop=True),
                train_y.reset_index(drop=True),
                test_window_df.reset_index(drop=True),
                selected,
                demographics,
                logistic_c=float(args.logistic_c),
                class_weight=args.class_weight,
                seed=int(args.seed) + fold_id * 1_000 + window_index * 10,
            )
            protein_coefficients.insert(0, "window", window)
            protein_coefficients.insert(0, "outer_fold", fold_id)
            coefficient_blocks.append(protein_coefficients)
            pred_frame = pd.DataFrame(
                {
                    "outer_fold": fold_id,
                    "panel_id": panel_id,
                    "candidate_pool_size": len(candidate_pool),
                    "panel_size": len(selected),
                    "window": window,
                    "window_label": train_qc["window_label"],
                    "participant_row": test_window_df[id_col].to_numpy(),
                    "target_y": test_y.to_numpy(dtype=int),
                    "selected_features": ";".join(selected),
                }
            )
            for model_key, values in predictions.items():
                pred_frame[f"p_{model_key}"] = values
            prediction_frames.append(pred_frame)

            for model_key, values in predictions.items():
                y = test_y.to_numpy(dtype=int)
                if len(np.unique(y)) < 2:
                    continue
                row = {
                    "outer_fold": fold_id,
                    "panel_id": panel_id,
                    "candidate_pool_size": len(candidate_pool),
                    "panel_size": len(selected),
                    "window": window,
                    "window_label": train_qc["window_label"],
                    "model_key": model_key,
                    "n": int(len(y)),
                    "cases": int(y.sum()),
                    "controls": int(len(y) - int(y.sum())),
                    "selected_features": ";".join(selected),
                }
                row.update(metric_values(y, values))
                split_metric_rows.append(row)

    if not prediction_frames:
        raise RuntimeError("No held-out predictions were generated.")
    pred_df = pd.concat(prediction_frames, ignore_index=True)
    split_metrics = pd.DataFrame(split_metric_rows)
    panel_df = pd.DataFrame(panel_rows)
    qc_df = pd.DataFrame(qc_rows)
    pooled = summarize_predictions(
        pred_df,
        split_metrics,
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    freq_df = feature_frequency(panel_df)
    outer_rankings = pd.concat(ranking_blocks, ignore_index=True)
    outer_integrated = pd.concat(integrated_blocks, ignore_index=True)
    stability_summary, stability_integrated = _aggregate_outer_training_stability(
        outer_rankings,
        list(args.windows),
        int(args.n_subsamples),
    )
    candidate_pool_df = pd.DataFrame(candidate_pool_rows)
    compression_df = pd.concat(compression_blocks, ignore_index=True)
    coefficient_df = pd.concat(coefficient_blocks, ignore_index=True)
    consensus_coefficients = _summarize_consensus_coefficients(
        coefficient_df,
        freq_df,
        windows=list(args.windows),
        n_folds=len(folds),
        panel_size=int(args.panel_size),
    )

    tables = {
        "outer_predictions.csv": pred_df,
        "outer_fold_window_model_metrics.csv": split_metrics,
        "outer_fold_candidate_pools.csv": candidate_pool_df,
        "outer_fold_panel_compression.csv": compression_df,
        "outer_fold_selected_panels.csv": panel_df,
        "outer_fold_window_protein_coefficients.csv": coefficient_df,
        "window_case_sparsity_qc.csv": qc_df,
        "pooled_metrics.csv": pooled,
        "panel_feature_frequency.csv": freq_df,
        "outer_fold_window_stability.csv": outer_rankings,
        "outer_fold_integrated_rankings.csv": outer_integrated,
        "fig4a_outer_training_stability_summary.csv": stability_summary,
        "fig4a_top20_integrated_ranking.csv": stability_integrated.head(20),
        "fig4c_consensus_coefficients.csv": consensus_coefficients,
    }
    for filename, table in tables.items():
        table.to_csv(out_dir / filename, index=False)
    _write_config(
        args,
        out_dir=out_dir,
        input_path=input_path,
        proteins_path=proteins_path,
        df=df,
        proteins=proteins,
        demographics=demographics,
        adjustment_covariates=adjustment_covariates,
        resolved_columns=resolved_columns,
    )
    _write_result_readme(out_dir, pooled, freq_df)
    plot_roc(pred_df, pooled, out_dir, panel_id="selected_panel")
    plot_consensus_coefficients(consensus_coefficients, out_dir, windows=list(args.windows))
    plot_metric_summary(pooled, out_dir)
    plot_feature_frequency(freq_df, out_dir, panel_id="selected_panel")
    print(f"Wrote outputs to {out_dir}", flush=True)
    return tables


def build_window(df: pd.DataFrame, window: str, control_mode: str):
    from .data import build_exact_window

    return build_exact_window(df, window, control_mode=control_mode)
