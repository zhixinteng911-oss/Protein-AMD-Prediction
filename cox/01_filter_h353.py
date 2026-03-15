#!/usr/bin/env python3
"""Restrict a broad H35 cohort to H35.3-specific AMD events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PARTICIPANT_COL = "Participant.ID"
ICD10_COL = "Diagnoses - ICD10"
H35_DATE_COL = "Date H35 first reported (other retinal disorders)"
BASELINE_DATE_COL = "Date of attending assessment centre | Instance 0"
DEATH_DATE_COL = "Date of death | Instance 0_y"
LOST_DATE_COL = "Date lost to follow-up_x"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", required=True, help="Broad H35 cohort CSV.")
    parser.add_argument("--output-file", required=True, help="Output CSV for H35.3-specific cohort.")
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
    return parser.parse_args()


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [
        column
        for column in [
            PARTICIPANT_COL,
            "target_y",
            ICD10_COL,
            H35_DATE_COL,
            BASELINE_DATE_COL,
            DEATH_DATE_COL,
            LOST_DATE_COL,
        ]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def build_h353_cohort(df: pd.DataFrame, censor_date: str) -> tuple[pd.DataFrame, dict]:
    working = df.copy()
    original_records = len(working)
    original_events = int(working["target_y"].fillna(0).sum())

    for column in [H35_DATE_COL, BASELINE_DATE_COL, DEATH_DATE_COL, LOST_DATE_COL]:
        working[column] = pd.to_datetime(working[column], errors="coerce")

    censor = pd.to_datetime(censor_date)
    working["censor_date"] = censor
    working["has_h353"] = (
        working[ICD10_COL]
        .astype(str)
        .str.contains(r"\bH35\.?3\b|H353", case=False, regex=True)
    )
    working["end_date"] = working[[H35_DATE_COL, DEATH_DATE_COL, LOST_DATE_COL, "censor_date"]].min(axis=1)
    working["target_y_h353"] = (
        (working["end_date"] == working[H35_DATE_COL])
        & working[H35_DATE_COL].notna()
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
        "required_columns": {
            "participant": PARTICIPANT_COL,
            "baseline_date": BASELINE_DATE_COL,
            "h35_date": H35_DATE_COL,
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

    df = load_input(input_path)
    filtered, summary = build_h353_cohort(df, args.censor_date)
    filtered.to_csv(output_path, index=False)
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved cohort: {output_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
