#!/usr/bin/env python3
"""Merge proteomics, H35.3 covariates, and PRS into the analysis-ready dataset."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CONTINUOUS_SPECS = {
    "f_21022": ("age_z", None),
    "f_21001_0": ("bmi_log_z", "log"),
    "f_22189": ("tdi_normal_z", "rankn"),
    "f_4080_0 | Array 0": ("sbp_z", None),
    "f_4079_0 | Array 0": ("dbp_z", None),
    "f_30750_0": ("hba1c_z", None),
    "prs_amd": ("prs_amd_z", None),
}

GENETIC_PC_COLUMNS = [f"Genetic principal components | Array {i}" for i in range(1, 21)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory used to construct default input/output paths.",
    )
    parser.add_argument("--prot-file", help="Proteomics CSV.")
    parser.add_argument("--cov-file", help="H35.3-specific covariate CSV.")
    parser.add_argument("--prs-file", help="PRS CSV.")
    parser.add_argument("--out-file", help="Output analysis-ready CSV.")
    parser.add_argument(
        "--censor-date",
        help="Administrative censoring date in YYYY-MM-DD format. Recorded in the data dictionary for provenance.",
    )
    parser.add_argument(
        "--data-dictionary-file",
        help="Markdown file describing the generated variables.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    data_dir = Path(args.data_dir).expanduser().resolve()
    defaults = {
        "prot_file": data_dir / "olink_data_imputed_normalized_enhanced.csv",
        "cov_file": data_dir / "amd_cox_analysis_ready_with_alcohol_h353_only.csv",
        "prs_file": data_dir / "PRS.csv",
        "out_file": data_dir / "amd_final_analysis_dataset_h353_only.csv",
        "data_dictionary_file": data_dir / "variable_documentation_h353.md",
    }
    resolved = {}
    for name, default in defaults.items():
        value = getattr(args, name)
        resolved[name] = Path(value).expanduser().resolve() if value else default
    resolved["out_file"].parent.mkdir(parents=True, exist_ok=True)
    resolved["data_dictionary_file"].parent.mkdir(parents=True, exist_ok=True)
    return resolved


def create_missingness_indicators(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    missing_stats: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in df.columns:
            continue
        indicator = f"{column}_missing"
        df[indicator] = df[column].isna().astype(int)
        missing_stats[column] = {
            "count": int(df[column].isna().sum()),
            "percent": float(df[column].isna().mean() * 100.0),
        }
    return df, missing_stats


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def process_continuous_features(df: pd.DataFrame) -> pd.DataFrame:
    for source, (target, transform) in CONTINUOUS_SPECS.items():
        if source not in df.columns:
            raise ValueError(f"Required continuous column missing: {source}")
        values = safe_numeric(df[source])
        values = values.fillna(values.median())
        if transform == "log":
            values = np.log(values + 1.0)
        elif transform == "rankn":
            ranks = stats.rankdata(values)
            values = pd.Series(stats.norm.ppf(ranks / (len(ranks) + 1)), index=df.index)
        std = values.std(ddof=0)
        if std == 0:
            raise ValueError(f"Column {source} has zero variance after preprocessing.")
        df[target] = (values - values.mean()) / std
    return df


def process_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    if "f_31" in df.columns:
        sex = df["f_31"].fillna(df["f_31"].mode().iloc[0]).astype(str)
        df["sex_binary"] = pd.to_numeric(
            sex.map(
                {
                    "Female": 0,
                    "female": 0,
                    "F": 0,
                    "f": 0,
                    "0": 0,
                    "0.0": 0,
                    "Male": 1,
                    "male": 1,
                    "M": 1,
                    "m": 1,
                    "1": 1,
                    "1.0": 1,
                }
            ),
            errors="coerce",
        ).fillna(0).astype(int)

    if "f_20116_0" in df.columns:
        smoke = df["f_20116_0"].fillna(df["f_20116_0"].mode().iloc[0])
        smoke_map = {
            "Never": 0,
            "never": 0,
            "No": 0,
            "no": 0,
            "0": 0,
            "0.0": 0,
            "Previous": 1,
            "previous": 1,
            "Former": 1,
            "former": 1,
            "Ex": 1,
            "ex": 1,
            "1": 1,
            "1.0": 1,
            "Current": 2,
            "current": 2,
            "Yes": 2,
            "yes": 2,
            "2": 2,
            "2.0": 2,
        }
        smoke_values = pd.to_numeric(smoke.astype(str).map(smoke_map), errors="coerce").fillna(0)
        df["smoker_current"] = (smoke_values == 2).astype(int)
        df["smoker_former"] = (smoke_values == 1).astype(int)

    if "f_1558_0" in df.columns:
        alcohol = df["f_1558_0"].fillna(df["f_1558_0"].mode().iloc[0])
        alcohol_map = {
            "Never": 0,
            "never": 0,
            "Special occasions only": 1,
            "special occasions only": 1,
            "Rarely": 1,
            "rarely": 1,
            "One to three times a month": 2,
            "Once or twice a week": 3,
            "Three or four times a week": 4,
            "Daily or almost daily": 5,
            "0": 0,
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
        }
        alcohol_values = pd.to_numeric(alcohol.astype(str).map(alcohol_map), errors="coerce").fillna(0)
        df["alcohol_frequent"] = (alcohol_values >= 3).astype(int)

    return df


def process_genetic_pcs(df: pd.DataFrame) -> pd.DataFrame:
    for index, column in enumerate(GENETIC_PC_COLUMNS, start=1):
        if column not in df.columns:
            raise ValueError(f"Required genetic PC column missing: {column}")
        df[f"genetic_pc{index}_filled"] = safe_numeric(df[column]).fillna(safe_numeric(df[column]).median())
    return df


def build_data_dictionary(missing_stats: dict[str, dict[str, float]], censor_date: str | None = None) -> str:
    lines = [
        "# AMD H35.3 Data Dictionary",
        "",
        f"Generated on: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
    ]
    if censor_date:
        lines.extend(
            [
                f"Administrative censor date: `{censor_date}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Core variables",
            "",
            "- `Participant.ID`: participant identifier",
            "- `target_y`: H35.3 event indicator",
            "- `BL2Target_yrs`: time from baseline to event/censoring in years",
            "",
            "## Processed continuous variables",
            "",
            "| Source | Output | Transform |",
            "| --- | --- | --- |",
        ]
    )
    for source, (target, transform) in CONTINUOUS_SPECS.items():
        lines.append(f"| `{source}` | `{target}` | `{transform or 'none'}` |")

    lines.extend(
        [
            "",
            "## Processed categorical variables",
            "",
            "- `sex_binary`: 0=female, 1=male",
            "- `smoker_current`: current smoker indicator",
            "- `smoker_former`: former smoker indicator",
            "- `alcohol_frequent`: drinking frequency >= 3 times per week",
            "",
            "## Genetic PCs",
            "",
            "- `genetic_pc1_filled` through `genetic_pc20_filled`: median-imputed genetic PCs",
            "",
            "## Missingness indicators",
            "",
        ]
    )
    for column, stats_dict in missing_stats.items():
        lines.append(
            f"- `{column}_missing`: {stats_dict['count']} missing values "
            f"({stats_dict['percent']:.2f}%) in the raw merged dataset"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    prot = pd.read_csv(paths["prot_file"])
    cov = pd.read_csv(paths["cov_file"]).rename(columns={"eid": "Participant.ID"})
    prs = pd.read_csv(paths["prs_file"])[
        ["Participant ID", "Standard PRS for age-related macular degeneration (AMD)"]
    ].rename(
        columns={
            "Participant ID": "Participant.ID",
            "Standard PRS for age-related macular degeneration (AMD)": "prs_amd",
        }
    )

    for frame, name in [(prot, "proteomics"), (cov, "covariates"), (prs, "prs")]:
        if "Participant.ID" not in frame.columns:
            raise ValueError(f"`Participant.ID` missing from {name} input.")

    merged = prot.merge(cov, on="Participant.ID", how="left").merge(prs, on="Participant.ID", how="left")
    required_core = ["Participant.ID", "target_y", "BL2Target_yrs"]
    missing_core = [column for column in required_core if column not in merged.columns]
    if missing_core:
        raise ValueError(f"Missing required merged columns: {missing_core}")

    variables_for_missingness = list(CONTINUOUS_SPECS) + ["f_31", "f_20116_0", "f_1558_0"] + GENETIC_PC_COLUMNS
    merged, missing_stats = create_missingness_indicators(merged, variables_for_missingness)
    merged = process_continuous_features(merged)
    merged = process_categorical_features(merged)
    merged = process_genetic_pcs(merged)

    merged.to_csv(paths["out_file"], index=False)
    paths["data_dictionary_file"].write_text(
        build_data_dictionary(missing_stats, censor_date=args.censor_date),
        encoding="utf-8",
    )

    print(f"saved dataset: {paths['out_file']}")
    print(f"saved data dictionary: {paths['data_dictionary_file']}")
    print(f"rows={len(merged)} cols={merged.shape[1]}")


if __name__ == "__main__":
    main()
