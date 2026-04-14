#!/usr/bin/env python3
"""Small shared helpers for the AMD ML scripts."""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile

if "MPLCONFIGDIR" not in os.environ:
    mpl_dir = Path(tempfile.gettempdir()) / "protein_amd_mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 2022
N_FOLDS = 5
PARTICIPANT_ID_COL = "Participant_ID"

WINDOW_CONFIG = {
    "within_3yr": {"title": "Within 3 years", "folder": "within_3yr"},
}

PREDICTION_INCLUDE_FUTURE_CASES = {
    "within_3yr": True,
}

COX_PROTEINS = [
    "acta2", "ada2", "adm", "apoa2", "b2m", "bcl2", "ccdc80", "ccl18", "ccl3", "cd14", "cd163",
    "cd300c", "cdcp1", "chga", "chrdl1", "ckb", "clec5a", "colec12", "cst3", "cstb", "ctsl", "cxcl17",
    "cxcl9", "dcbld2", "dkk3", "dsc2", "ebi3_il27", "eda2r", "efemp1", "eln", "enpp5", "epha2", "fga",
    "fgl1", "gdf15", "gfap", "havcr2", "ifi30", "igdcc4", "igfbp2", "igfbp4", "il15ra", "il18bp", "il6",
    "itih3", "klk4", "lair1", "layn", "lect2", "lgals9", "lmnb2", "lpcat2", "lrrn1", "ltbp2", "mmp12",
    "mmp7", "msr1", "nectin2", "nefl", "nt5c1a", "ntprobnp", "pga4", "pilra", "pilrb", "plaur", "psg1",
    "ret", "rnase1", "scarb2", "scarf2", "septin8", "serpina3", "spink1", "spp1", "tff1", "tff2", "tff3",
    "tgfbr2", "timp1", "timp4", "tnfrsf10a", "tnfrsf10b", "tnfrsf11a", "tnfrsf11b", "tnfrsf1a", "tnfrsf1b",
    "trem2", "vsig2", "vsig4", "wfdc2", "wfikkn1", "yap1",
]

DEMO_COLS = [
    "Age_at_recruitment",
    "sex_binary",
    "bmi_log_z",
    "smoker_current",
    "ldlr",
    "alcohol_frequent",
]

DEMO_COL_ALIASES = {
    "Age_at_recruitment": ["Age_at_recruitment", "Age at recruitment"],
    "sex_binary": ["sex_binary"],
    "bmi_log_z": ["bmi_log_z"],
    "smoker_current": ["smoker_current"],
    "ldlr": ["ldlr"],
    "alcohol_frequent": ["alcohol_frequent"],
}

FEATURE_DISPLAY_ALIASES = {
    "ebi3_il27": "IL27",
}

ROC_PLOT_STYLE = {
    "figsize": (6.0, 6.0),
    "line_width": 1.5,
    "diag_width": 1.0,
    "curve_mode": "plot",
    "roc_drop_intermediate": False,
    "xlabel_size": 14,
    "ylabel_size": 14,
    "tick_size": 12,
    "label_text_size": 12,
    "legend_size": 10,
    "legend_facecolor": "#b3b3b3",
    "legend_alpha": 0.8,
    "legend_edgecolor": "gray",
    "legend_fancybox": False,
}

LGBM_PROTEIN_PARAMS = {
    "n_estimators": 500,
    "num_leaves": 7,
    "max_depth": 19,
    "subsample": 0.65,
    "colsample_bytree": 0.55,
    "reg_alpha": 0.12011365907981926,
    "reg_lambda": 22.185215393398,
    "scale_pos_weight": 11.581156617386851,
    "min_child_samples": 17,
    "learning_rate": 0.01,
    "random_state": RANDOM_STATE,
    "verbose": -1,
    "device": "cpu",
    "n_jobs": 1,
}

LGBM_DEMO_PARAMS = {
    **LGBM_PROTEIN_PARAMS,
    "colsample_bytree": 1.0,
    "num_leaves": 5,
    "max_depth": 5,
}

LGBM_COMBINED_PARAMS = {
    **LGBM_PROTEIN_PARAMS,
    "colsample_bytree": 0.85,
    "subsample": 0.80,
    "num_leaves": 10,
    "n_estimators": 600,
}


def get_output_dir(base_out: str | Path, window: str) -> Path:
    out_dir = Path(base_out).expanduser().resolve() / WINDOW_CONFIG[window]["folder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_prediction_include_future_cases(window: str) -> bool:
    return PREDICTION_INCLUDE_FUTURE_CASES.get(window, False)


def sanitize(col: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", str(col)).strip("_")


def find_optimal_k(s04_df: pd.DataFrame) -> int:
    return int(s04_df["AUC_mean"].idxmax()) + 1


def load_dataframe(input_file: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(input_file).expanduser().resolve(), low_memory=False)
    df = df.rename(columns={col: sanitize(col) for col in df.columns})
    df["target_y"] = pd.to_numeric(df["target_y"], errors="coerce").fillna(0).astype(int)
    df["BL2Target_yrs"] = pd.to_numeric(df["BL2Target_yrs"], errors="coerce")
    return df.reset_index(drop=True)


def build_demo_frame(window_df: pd.DataFrame) -> pd.DataFrame:
    demo_series = {}
    for canonical_name in DEMO_COLS:
        for candidate in DEMO_COL_ALIASES.get(canonical_name, [canonical_name]):
            sanitized_candidate = sanitize(candidate)
            if candidate in window_df.columns:
                demo_series[canonical_name] = window_df[candidate]
                break
            if sanitized_candidate in window_df.columns:
                demo_series[canonical_name] = window_df[sanitized_candidate]
                break

    x_demo = pd.DataFrame(demo_series).reset_index(drop=True)
    for col in x_demo.columns:
        if x_demo[col].dtype == "object":
            x_demo[col] = x_demo[col].astype("category").cat.codes
        else:
            x_demo[col] = pd.to_numeric(x_demo[col], errors="coerce")
    return x_demo


def feature_display_name(feature: str) -> str:
    return FEATURE_DISPLAY_ALIASES.get(feature, feature.upper())


def get_roc_plot_style() -> dict[str, object]:
    return dict(ROC_PLOT_STYLE)


def build_window_dataset(
    df: pd.DataFrame,
    window: str,
    *,
    include_future_cases: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    years = pd.to_numeric(df["BL2Target_yrs"], errors="coerce")
    target = pd.to_numeric(df["target_y"], errors="coerce").fillna(0).astype(int)

    if window == "within_3yr":
        y = target.copy()
        y.loc[years > 3] = 0
        if include_future_cases:
            valid = pd.Series(True, index=df.index)
        else:
            mask_exclude = (target == 1) & (years > 3)
            valid = ~mask_exclude
    else:
        raise ValueError(f"Unknown window: {window}")

    window_df = df.loc[valid].reset_index(drop=True)
    window_y = y.loc[valid].reset_index(drop=True)
    return window_df, window_y


def prepare_payload(
    input_file: str | Path,
    window: str,
    *,
    include_future_cases: bool = False,
) -> dict[str, object]:
    df = load_dataframe(input_file)
    window_df, y = build_window_dataset(df, window, include_future_cases=include_future_cases)
    protein_cols = [col for col in COX_PROTEINS if col in window_df.columns]
    x_protein = window_df[protein_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    x_demo = build_demo_frame(window_df)
    demo_cols = list(x_demo.columns)

    x_all = pd.concat([x_protein, x_demo], axis=1).reset_index(drop=True)
    cross_cols: list[str] = []
    age_col = "Age_at_recruitment"
    if age_col in x_all.columns:
        top5_prots = [prot for prot in COX_PROTEINS if prot in x_all.columns][:5]
        for prot in top5_prots:
            cross_col = f"{prot}_x_age"
            x_all[cross_col] = x_all[prot] * x_all[age_col]
            cross_cols.append(cross_col)

    return {
        "df": window_df,
        "y": y.reset_index(drop=True).astype(int),
        "X_all": x_all,
        "X_protein": x_protein,
        "X_demo": x_demo,
        "protein_cols": protein_cols,
        "demo_cols": demo_cols,
        "cross_cols": cross_cols,
    }


def prepare_prediction_payload(
    input_file: str | Path,
    window: str,
) -> dict[str, object]:
    return prepare_payload(
        input_file,
        window,
        include_future_cases=get_prediction_include_future_cases(window),
    )


def make_folds(
    y: pd.Series,
    requested_splits: int = N_FOLDS,
    *,
    strict: bool = False,
    label: str = "cross-validation",
) -> list[tuple[np.ndarray, np.ndarray]]:
    positives = int(y.sum())
    negatives = int((1 - y).sum())
    max_splits = min(requested_splits, positives, negatives)
    if strict and max_splits < requested_splits:
        raise ValueError(
            f"{label} requested {requested_splits} folds but only {positives} cases and {negatives} controls are available."
        )
    if max_splits < 2:
        raise ValueError(f"{label} requires at least 2 folds; got {positives} cases and {negatives} controls.")
    splitter = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=RANDOM_STATE)
    dummy = np.zeros(len(y))
    return list(splitter.split(dummy, y.to_numpy(dtype=int)))


def fit_model(x_train: pd.DataFrame, y_train: np.ndarray, params: dict) -> tuple[LGBMClassifier, pd.Series]:
    train_means = x_train.mean()
    model = LGBMClassifier(**params)
    model.fit(x_train.fillna(train_means), y_train)
    return model, train_means


def predict_model(model: LGBMClassifier, x_test: pd.DataFrame, means: pd.Series) -> np.ndarray:
    return model.predict_proba(x_test.fillna(means))[:, 1]


def rank_features_elasticnet_cv(
    x_protein: pd.DataFrame,
    y: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    *,
    random_state: int = RANDOM_STATE,
    c: float = 0.1,
    l1_ratio: float = 0.5,
    max_iter: int = 4000,
) -> pd.DataFrame:
    rank_res = []
    for train_idx, _ in folds:
        x_train = x_protein.iloc[train_idx]
        y_train = y.iloc[train_idx].to_numpy(dtype=int)
        train_means = x_train.mean()
        x_train_filled = x_train.fillna(train_means).fillna(0)
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="elasticnet",
                        solver="saga",
                        l1_ratio=l1_ratio,
                        C=c,
                        max_iter=max_iter,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        model.fit(x_train_filled, y_train)
        rank_res.append(
            pd.Series(
                np.abs(model.named_steps["clf"].coef_).ravel(),
                index=x_protein.columns,
            )
        )

    avg_coef = pd.concat(rank_res, axis=1).mean(axis=1).sort_values(ascending=False)
    rank_df = avg_coef.reset_index()
    rank_df.columns = ["Features", "Gain"]
    max_gain = float(rank_df["Gain"].max()) if len(rank_df) else 0.0
    rank_df["Cover"] = rank_df["Gain"] / max_gain if max_gain > 0 else 0.0
    return rank_df
