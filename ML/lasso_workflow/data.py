from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .config import (
    ADJUSTMENT_ALIASES,
    DEMO_ALIASES,
    DEMO_COLS,
    ID_ALIASES,
    OUTCOME_ALIASES,
    WINDOW_SPECS,
)


def sanitize(value: object) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")


def normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def resolve_path(path_text: str | Path, script_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.exists() or path.is_absolute():
        return path
    fallback = script_dir / path
    return fallback if fallback.exists() else path


def _lookup(columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for col in columns:
        out.setdefault(normalize_key(col), col)
    return out


def _first_present(lookup: dict[str, str], aliases: list[str]) -> str | None:
    for alias in aliases:
        col = lookup.get(normalize_key(alias))
        if col is not None:
            return col
    return None


def load_candidate_proteins(path: str | Path) -> list[str]:
    manifest = pd.read_csv(path)
    if "protein" not in manifest.columns:
        raise ValueError("Protein manifest must contain a 'protein' column.")
    proteins = [
        sanitize(value).lower()
        for value in manifest["protein"].dropna().astype(str)
        if str(value).strip()
    ]
    proteins = list(dict.fromkeys(proteins))
    if not proteins:
        raise ValueError("Protein manifest is empty.")
    return proteins


def _read_selected_columns(input_path: Path, raw_columns: list[str]) -> pd.DataFrame:
    raw_columns = list(dict.fromkeys(raw_columns))
    return pd.read_csv(input_path, usecols=raw_columns, low_memory=False)


def read_analysis_frame(
    input_file: str | Path,
    *,
    proteins_file: str | Path,
    id_col: str | None,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], str, dict[str, str]]:
    input_path = Path(input_file).expanduser().resolve()
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    lookup = _lookup(header)
    requested_proteins = load_candidate_proteins(proteins_file)

    target_raw = _first_present(lookup, OUTCOME_ALIASES["target_y"])
    time_raw = _first_present(lookup, OUTCOME_ALIASES["BL2Target_yrs"])
    if target_raw is None or time_raw is None:
        raise ValueError(
            "Input must contain outcome/time columns: target_y or incident_amd, "
            "and BL2Target_yrs or years_to_amd."
        )

    id_raw = _first_present(lookup, [id_col] if id_col else [])
    if id_raw is None:
        id_raw = _first_present(lookup, ID_ALIASES)
    if id_raw is None:
        raise ValueError("A participant identifier column is required.")

    protein_raw_by_clean: dict[str, str] = {}
    missing_proteins: list[str] = []
    for protein in requested_proteins:
        raw = lookup.get(normalize_key(protein))
        if raw is None:
            missing_proteins.append(protein)
        else:
            protein_raw_by_clean[protein] = raw
    if missing_proteins:
        raise ValueError(
            "Candidate proteins not found in input: " + ", ".join(missing_proteins)
        )

    demo_raw_by_clean: dict[str, str] = {}
    for canonical in DEMO_COLS:
        raw = _first_present(lookup, DEMO_ALIASES[canonical])
        if raw is not None:
            demo_raw_by_clean[canonical] = raw

    adjustment_raw_by_clean: dict[str, str] = {}
    missing_adjustment: list[str] = []
    for canonical, aliases in ADJUSTMENT_ALIASES.items():
        raw = _first_present(lookup, aliases)
        if raw is None:
            missing_adjustment.append(canonical)
        else:
            adjustment_raw_by_clean[canonical] = raw
    if missing_adjustment:
        raise ValueError(
            "Missing required stability-adjustment covariates: "
            + ", ".join(missing_adjustment)
        )

    raw_cols = [target_raw, time_raw]
    raw_cols.append(id_raw)
    raw_cols.extend(protein_raw_by_clean.values())
    raw_cols.extend(demo_raw_by_clean.values())
    raw_cols.extend(adjustment_raw_by_clean.values())
    raw = _read_selected_columns(input_path, raw_cols)

    rename: dict[str, str] = {
        target_raw: "target_y",
        time_raw: "BL2Target_yrs",
    }
    rename[id_raw] = "source_participant_id"
    for clean, raw_name in protein_raw_by_clean.items():
        rename[raw_name] = clean
    for clean, raw_name in demo_raw_by_clean.items():
        rename[raw_name] = clean
    for clean, raw_name in adjustment_raw_by_clean.items():
        rename[raw_name] = clean

    df = raw.rename(columns=rename).copy()
    target = pd.to_numeric(df["target_y"], errors="coerce")
    years = pd.to_numeric(df["BL2Target_yrs"], errors="coerce")
    if not set(target.dropna().unique()).issubset({0, 1}):
        raise ValueError("target_y / incident_amd must contain only 0/1 values.")
    eligible = target.isin([0, 1]) & years.notna() & years.gt(0)
    df = df.loc[eligible].copy()
    if df.empty:
        raise ValueError("No participants have valid outcome and follow-up data.")
    df["target_y"] = target.loc[eligible].astype(int)
    df["BL2Target_yrs"] = years.loc[eligible]
    if df["source_participant_id"].isna().any():
        raise ValueError("Participant identifiers must not be missing.")
    if df["source_participant_id"].duplicated().any():
        raise ValueError("Participant identifiers must be unique.")
    df["row_id"] = np.arange(len(df), dtype=int)
    proteins = list(protein_raw_by_clean.keys())
    demo_cols = [col for col in DEMO_COLS if col in df.columns]
    adjustment_cols = [col for col in ADJUSTMENT_ALIASES if col in df.columns]
    resolved = {
        "target_y": target_raw,
        "BL2Target_yrs": time_raw,
        "participant_id": id_raw,
        "row_id": "row_id",
        **{f"protein:{k}": v for k, v in protein_raw_by_clean.items()},
        **{f"demographic:{k}": v for k, v in demo_raw_by_clean.items()},
        **{f"adjustment:{k}": v for k, v in adjustment_raw_by_clean.items()},
    }
    return df.reset_index(drop=True), proteins, demo_cols, adjustment_cols, "row_id", resolved


def build_exact_window(
    df: pd.DataFrame,
    window: str,
    *,
    control_mode: str,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    if window not in WINDOW_SPECS:
        raise ValueError(f"Unknown temporal window: {window}")
    spec = WINDOW_SPECS[window]
    years = pd.to_numeric(df["BL2Target_yrs"], errors="coerce")
    target = pd.to_numeric(df["target_y"], errors="coerce").fillna(0).astype(int)
    valid_time = years.notna() & years.gt(0)

    case_min = float(spec["case_min"])
    case_max = spec["case_max"]
    if case_max is None:
        case_mask = valid_time & target.eq(1) & years.gt(case_min)
    else:
        case_mask = valid_time & target.eq(1) & years.gt(case_min) & years.le(float(case_max))

    control_min = float(spec["control_followup_min"])
    event_free_controls = valid_time & target.eq(0) & years.ge(control_min)
    if control_mode == "event_free":
        control_mask = event_free_controls
    elif control_mode == "horizon":
        if case_max is None:
            control_mask = event_free_controls
        else:
            later_cases = valid_time & target.eq(1) & years.gt(float(case_max))
            control_mask = event_free_controls | later_cases
    else:
        raise ValueError("--control-mode must be event_free or horizon.")

    selected = case_mask | control_mask
    y = pd.Series(0, index=df.index, dtype=int)
    y.loc[case_mask] = 1
    window_df = df.loc[selected].reset_index(drop=True)
    window_y = y.loc[selected].reset_index(drop=True)
    qc = {
        "window": window,
        "window_label": spec["label"],
        "control_mode": control_mode,
        "n": int(len(window_y)),
        "cases": int(window_y.sum()),
        "controls": int(len(window_y) - int(window_y.sum())),
        "case_time_min_exclusive": case_min,
        "case_time_max_inclusive": case_max,
        "control_followup_min": control_min,
    }
    return window_df, window_y, qc


def coerce_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in out.columns:
        values = pd.to_numeric(out[col], errors="coerce")
        if values.notna().any():
            out[col] = values
        else:
            codes = out[col].astype("category").cat.codes.astype(float)
            out[col] = codes.where(codes >= 0, np.nan)
    return out
