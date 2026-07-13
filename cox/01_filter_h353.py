#!/usr/bin/env python3
"""Restrict a broad H35 cohort to H35.3-specific AMD events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PARTICIPANT_COL = "Participant.ID"
ICD10_COL = "Diagnoses - ICD10"
H353_DATE_FIELD_ID = "131182"
H353_EVENT_DATE_COL = "h353_first_reported_date"
H353_DATE_COL_CANDIDATES = [
    "f.131182.0.0",
    "f_131182_0_0",
    "131182-0.0",
    "131182",
    "Date H35.3 first reported",
    "Date H35.3 first reported (field 131182)",
]
BASELINE_DATE_COL = "Date of attending assessment centre | Instance 0"
DEATH_DATE_COL = "Date of death | Instance 0_y"
LOST_DATE_COL = "Date lost to follow-up_x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="Broad H35 cohort CSV.")
    parser.add_argument(
        "--output-file",
        required=True,
        help="Output CSV for H35.3-specific cohort.",
    )
    parser.add_argument(
        "--log-file",
        required=True,
        help="JSON summary file describing event counts and filtering decisions.",
    )
    parser.add_argument(
        "--censor-date",
        required=True,
        help="Administrative censoring date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--h353-date-col",
        help=(
            "Column containing the H35.3 first-reported date "
            f"(UKB field {H353_DATE_FIELD_ID}). If omitted, common field-"
            "based column names are searched."
        ),
    )
    return parser.parse_args()


def resolve_h353_date_col(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested H35.3 date column not found: {requested}")
        return requested

    for column in H353_DATE_COL_CANDIDATES:
        if column in df.columns:
            return column

    field_matches = [column for column in df.columns if H353_DATE_FIELD_ID in column]
    if len(field_matches) == 1:
        return field_matches[0]
    if len(field_matches) > 1:
        raise ValueError(
            "Multiple columns contain UKB field 131182. Pass --h353-date-col "
            f"explicitly. Matches: {field_matches}"
        )

    raise ValueError(
        "No H35.3 first-reported date column was found. Provide the UKB field "
        "131182 date column with --h353-date-col."
    )


def load_input(path: Path, h353_date_col: str | None) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path, low_memory=False)
    resolved_date_col = resolve_h353_date_col(df, h353_date_col)
    missing = [
        column
        for column in [
            PARTICIPANT_COL,
            "target_y",
            ICD10_COL,
            BASELINE_DATE_COL,
            DEATH_DATE_COL,
            LOST_DATE_COL,
        ]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if H353_EVENT_DATE_COL in df.columns and resolved_date_col != H353_EVENT_DATE_COL:
        raise ValueError(
            f"Input already contains `{H353_EVENT_DATE_COL}` and also "
            f"`{resolved_date_col}`. Pass --h353-date-col explicitly after "
            "removing or renaming the duplicate date column."
        )
    if resolved_date_col != H353_EVENT_DATE_COL:
        df = df.rename(columns={resolved_date_col: H353_EVENT_DATE_COL})
    return df, resolved_date_col


def build_h353_cohort(
    df: pd.DataFrame,
    censor_date: str,
    source_h353_date_col: str,
) -> tuple[pd.DataFrame, dict]:
    working = df.copy()
    original_records = len(working)
    original_events = int(working["target_y"].fillna(0).sum())

    for column in [H353_EVENT_DATE_COL, BASELINE_DATE_COL, DEATH_DATE_COL, LOST_DATE_COL]:
        working[column] = pd.to_datetime(working[column], errors="coerce")

    censor = pd.to_datetime(censor_date)
    working["censor_date"] = censor
    working["has_h353"] = (
        working[ICD10_COL]
        .astype(str)
        .str.contains(r"\bH35\.?3\b|H353", case=False, regex=True)
    )
    working["end_date"] = working[
        [H353_EVENT_DATE_COL, DEATH_DATE_COL, LOST_DATE_COL, "censor_date"]
    ].min(axis=1)
    working["target_y_h353"] = (
        (working["end_date"] == working[H353_EVENT_DATE_COL])
        & working[H353_EVENT_DATE_COL].notna()
        & working["has_h353"]
    ).astype(int)
    working["target_y_original"] = working["target_y"]
    working["target_y"] = working["target_y_h353"]
    working["BL2Target_yrs"] = (working["end_date"] - working[BASELINE_DATE_COL]).dt.days / 365.25

    filtered = working.loc[working["BL2Target_yrs"] > 0].copy()
    filtered.drop(columns=["has_h353", "target_y_h353"], inplace=True)

    summary = {
        "input_records": original_records,
        "input_events_all_h35": original_events,
        "output_records": int(len(filtered)),
        "output_events_h353": int(filtered["target_y"].sum()),
        "non_h353_events_removed": int(original_events - filtered["target_y"].sum()),
        "censor_date": censor_date,
        "event_definition": "H35.3-restricted",
        "h353_date_field_id": H353_DATE_FIELD_ID,
        "h353_date_source_column": source_h353_date_col,
        "h353_date_analysis_column": H353_EVENT_DATE_COL,
        "required_columns": {
            "participant": PARTICIPANT_COL,
            "baseline_date": BASELINE_DATE_COL,
            "h353_first_reported_date": H353_EVENT_DATE_COL,
            "diagnosis": ICD10_COL,
        },
    }
    return filtered, summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    output_path = Path(args.output_file).expanduser().resolve()
    log_path = Path(args.log_file).expanduser().resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    df, source_h353_date_col = load_input(input_path, args.h353_date_col)
    filtered, summary = build_h353_cohort(df, args.censor_date, source_h353_date_col)
    filtered.to_csv(output_path, index=False)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved cohort: {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
