#!/usr/bin/env python3
"""Protein-wise Cox regression for the H35.3-specific AMD cohort."""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


TIME_COL = "BL2Target_yrs"
EVENT_COL = "target_y"
M2_COVS = ["age_z", "sex_binary"]
M3_COVS = [
    "age_z",
    "bmi_log_z",
    "tdi_normal_z",
    "sbp_z",
    "dbp_z",
    "hba1c_z",
    "prs_amd_z",
    "sex_binary",
    "smoker_current",
    "smoker_former",
    "alcohol_frequent",
] + [f"genetic_pc{i}_filled" for i in range(1, 21)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, help="Analysis-ready dataset CSV.")
    parser.add_argument("--protein-list-file", required=True, help="CSV whose first row contains protein names.")
    parser.add_argument("--out-dir", required=True, help="Directory for result files.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=min(10, os.cpu_count() or 1),
        help="Number of parallel workers.",
    )
    parser.add_argument("--batch-size", type=int, default=25, help="Proteins per checkpoint batch.")
    parser.add_argument("--debug", action="store_true", help="Restrict analysis to the first 5000 rows and 5 proteins.")
    return parser.parse_args()


def read_protein_list(path: Path) -> list[str]:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    return [token.strip() for token in first_line.split(",") if token.strip()]


def load_data(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    df = pd.read_csv(args.data_file)
    if args.debug:
        df = df.head(5000).copy()

    required = [TIME_COL, EVENT_COL]
    missing_required = [column for column in required if column not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in dataset: {missing_required}")

    protein_names = read_protein_list(Path(args.protein_list_file))
    proteins = [column for column in df.columns if column in protein_names]
    if args.debug:
        proteins = proteins[:5]
    if not proteins:
        raise ValueError("No protein columns overlap between the dataset and the protein list file.")

    missing_m2 = [column for column in M2_COVS if column not in df.columns]
    if missing_m2:
        raise ValueError(f"Missing required Model 2 covariates: {missing_m2}")

    m3_missing = [column for column in M3_COVS if column not in df.columns]
    if m3_missing:
        raise ValueError(f"Missing required Model 3 covariates: {m3_missing}")
    return df, proteins, M2_COVS, M3_COVS, m3_missing


def fit_cox(df_fit: pd.DataFrame, protein: str) -> tuple[dict, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df_fit, duration_col=TIME_COL, event_col=EVENT_COL)
    warning_messages = [str(item.message) for item in caught]
    ci = np.exp(cph.confidence_intervals_.loc[protein])
    result = {
        "hr": float(np.exp(cph.params_[protein])),
        "hr_lower_ci": float(ci.iloc[0]),
        "hr_upper_ci": float(ci.iloc[1]),
        "p_value": float(cph.summary.loc[protein, "p"]),
    }
    return result, warning_messages


def prepare_model_frame(
    data: pd.DataFrame,
    protein: str,
    covariates: list[str],
) -> pd.DataFrame:
    cols = [TIME_COL, EVENT_COL, protein] + covariates
    return data[cols].dropna().copy()


def analyse_protein(
    protein: str,
    data: pd.DataFrame,
    m2_covs: list[str],
    m3_covs: list[str],
) -> tuple[list[dict], list[dict], list[dict]]:
    models = [
        ("Model 1 - Unadjusted", []),
        ("Model 2 - Age + Sex", m2_covs),
        ("Model 3 - Fully Adjusted", m3_covs),
    ]

    results: list[dict] = []
    warning_rows: list[dict] = []
    exclusion_rows: list[dict] = []
    for model_name, covariates in models:
        df_model = prepare_model_frame(data, protein, covariates)
        n_samples = int(len(df_model))
        n_events = int(df_model[EVENT_COL].sum()) if n_samples else 0
        if n_samples < 50 or n_events < 5:
            exclusion_rows.append(
                {
                    "protein": protein,
                    "model": model_name,
                    "n_samples_complete_case": n_samples,
                    "n_events_complete_case": n_events,
                    "exclusion_reason": "insufficient_complete_case_samples_or_events",
                }
            )
            continue

        try:
            fit_result, warning_messages = fit_cox(df_model, protein)
        except Exception as exc:  # pragma: no cover - numerical failures are data dependent
            warning_rows.append(
                {
                    "protein": protein,
                    "model": model_name,
                    "warning_type": "fit_error",
                    "message": str(exc),
                }
            )
            continue

        results.append(
            {
                "protein": protein,
                "model": model_name,
                "n_samples": n_samples,
                "n_events": n_events,
                "covariates": "; ".join(covariates) if covariates else "None",
                "n_covariates": len(covariates),
                **fit_result,
            }
        )
        for message in warning_messages:
            warning_rows.append(
                {
                    "protein": protein,
                    "model": model_name,
                    "warning_type": "fit_warning",
                    "message": message,
                }
            )

    return results, warning_rows, exclusion_rows


def apply_multiple_testing(results_df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for model_name, group in results_df.groupby("model"):
        group = group.copy()
        p_values = group["p_value"].dropna()
        if p_values.empty:
            parts.append(group)
            continue
        _, fdr, _, _ = multipletests(p_values, method="fdr_bh")
        _, bonferroni, _, _ = multipletests(p_values, method="bonferroni")
        mask = group["p_value"].notna()
        group.loc[mask, "fdr_p"] = fdr
        group.loc[mask, "bonferroni_p"] = bonferroni
        group.loc[:, "model_name"] = model_name
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data, proteins, m2_covs, m3_covs, m3_missing = load_data(args)
    metadata = {
        "data_file": str(Path(args.data_file).expanduser().resolve()),
        "protein_list_file": str(Path(args.protein_list_file).expanduser().resolve()),
        "n_rows": int(len(data)),
        "n_events": int(data[EVENT_COL].sum()),
        "n_proteins_requested": len(proteins),
        "m2_covariates": m2_covs,
        "m3_covariates_available": m3_covs,
        "m3_covariates_missing": m3_missing,
        "n_jobs": args.n_jobs,
        "batch_size": args.batch_size,
        "debug": args.debug,
    }

    all_results: list[dict] = []
    all_warnings: list[dict] = []
    excluded_proteins: list[dict] = []
    batches = [proteins[index : index + args.batch_size] for index in range(0, len(proteins), args.batch_size)]

    for batch_index, batch in enumerate(batches, start=1):
        batch_payload = Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(analyse_protein)(protein, data, m2_covs, m3_covs)
            for protein in tqdm(batch, desc=f"batch {batch_index}/{len(batches)}", leave=False)
        )
        batch_results = [result for results, _warnings, _excluded in batch_payload for result in results]
        batch_warnings = [warning for _results, warnings_list, _excluded in batch_payload for warning in warnings_list]
        batch_excluded = [row for _results, _warnings, excluded_rows in batch_payload for row in excluded_rows]
        all_results.extend(batch_results)
        all_warnings.extend(batch_warnings)
        excluded_proteins.extend(batch_excluded)

        if batch_results:
            pd.DataFrame(batch_results).to_csv(out_dir / f"cox_h353_batch{batch_index:02d}.csv", index=False)

    if not all_results:
        raise RuntimeError("No Cox results were produced. Check event counts and input columns.")

    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv(out_dir / "cox_h353_raw.csv", index=False)

    adjusted_df = apply_multiple_testing(raw_df)
    adjusted_df.to_csv(out_dir / "cox_h353_adjusted.csv", index=False)

    warnings_df = pd.DataFrame(all_warnings)
    warnings_df.to_csv(out_dir / "cox_h353_warnings.csv", index=False)

    excluded_df = pd.DataFrame(excluded_proteins)
    excluded_df.to_csv(out_dir / "cox_h353_excluded_proteins.csv", index=False)

    metadata["n_proteins_analyzed"] = int(raw_df["protein"].nunique())
    metadata["n_proteins_excluded"] = int(excluded_df["protein"].nunique()) if not excluded_df.empty else 0

    (out_dir / "cox_h353_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"saved raw results: {out_dir / 'cox_h353_raw.csv'}")
    print(f"saved adjusted results: {out_dir / 'cox_h353_adjusted.csv'}")
    print(f"saved warnings: {out_dir / 'cox_h353_warnings.csv'}")
    print(f"saved exclusions: {out_dir / 'cox_h353_excluded_proteins.csv'}")


if __name__ == "__main__":
    main()
