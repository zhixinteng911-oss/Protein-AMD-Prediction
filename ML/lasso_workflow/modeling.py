from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import MODEL_LABELS, WINDOWS
from .data import build_exact_window, coerce_model_frame


@dataclass(frozen=True)
class SelectionConfig:
    n_subsamples: int
    sample_frac: float
    control_case_ratio: float
    lasso_c: float
    lasso_max_iter: int
    lasso_tol: float
    class_weight: str
    seed: int


def feature_display_name(feature: str) -> str:
    return {"ebi3_il27": "IL27"}.get(str(feature), str(feature).upper())


def make_outer_folds(df: pd.DataFrame, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    target = pd.to_numeric(df["target_y"], errors="raise").astype(int)
    years = pd.to_numeric(df["BL2Target_yrs"], errors="raise")
    strata = pd.Series("control", index=df.index, dtype="object")
    strata.loc[target.eq(1) & years.le(3)] = "case_0_3"
    strata.loc[target.eq(1) & years.gt(3) & years.le(9)] = "case_3_9"
    strata.loc[target.eq(1) & years.gt(9)] = "case_gt9"
    counts = strata.value_counts()
    if (counts < int(n_splits)).any():
        sparse = ", ".join(f"{name}={count}" for name, count in counts[counts < int(n_splits)].items())
        raise ValueError(f"Each outer-fold stratum needs at least {n_splits} participants: {sparse}")
    cv = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    return list(cv.split(np.zeros(len(strata)), strata.to_numpy()))


def _class_weight_arg(value: str) -> str | None:
    return None if value == "none" else value


def _fit_sparse_lasso(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    c: float,
    class_weight: str,
    max_iter: int,
    tol: float,
    seed: int,
) -> np.ndarray:
    x = coerce_model_frame(x)
    empty_columns = x.columns[x.isna().all()]
    if len(empty_columns):
        x.loc[:, empty_columns] = 0.0
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=float(c),
                    class_weight=_class_weight_arg(class_weight),
                    max_iter=int(max_iter),
                    tol=float(tol),
                    random_state=int(seed),
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(x, y.to_numpy(dtype=int))
    return model.named_steps["clf"].coef_.ravel()


def _stratified_subsample_indices(
    y: pd.Series,
    *,
    frac: float,
    control_case_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    y_arr = y.to_numpy(dtype=int)
    case_idx = np.flatnonzero(y_arr == 1)
    control_idx = np.flatnonzero(y_arr == 0)
    if len(case_idx) < 2 or len(control_idx) < 2:
        raise ValueError("Stability selection requires at least two cases and two controls.")
    n_cases = min(len(case_idx), max(1, int(round(len(case_idx) * float(frac)))))
    sampled_cases = rng.choice(case_idx, size=n_cases, replace=False)
    if float(control_case_ratio) > 0:
        n_controls = min(
            len(control_idx),
            max(1, int(round(n_cases * float(control_case_ratio)))),
        )
    else:
        n_controls = min(len(control_idx), max(1, int(round(len(control_idx) * float(frac)))))
    sampled_controls = rng.choice(control_idx, size=n_controls, replace=False)
    return np.sort(np.concatenate([sampled_cases, sampled_controls]))


def run_window_stability_train_only(
    train_df: pd.DataFrame,
    proteins: list[str],
    adjustment_features: list[str],
    window: str,
    *,
    control_mode: str,
    config: SelectionConfig,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    window_df, y, qc = build_exact_window(train_df, window, control_mode=control_mode)
    model_features = list(dict.fromkeys(proteins + adjustment_features))
    x = coerce_model_frame(window_df[model_features]).reset_index(drop=True)
    y = y.reset_index(drop=True).astype(int)
    rng = np.random.default_rng(int(seed))
    coef_blocks: list[pd.Series] = []
    for i in range(int(config.n_subsamples)):
        idx = _stratified_subsample_indices(
            y,
            frac=float(config.sample_frac),
            control_case_ratio=float(config.control_case_ratio),
            rng=rng,
        )
        coef = _fit_sparse_lasso(
            x.iloc[idx].reset_index(drop=True),
            y.iloc[idx].reset_index(drop=True),
            c=float(config.lasso_c),
            class_weight=config.class_weight,
            max_iter=int(config.lasso_max_iter),
            tol=float(config.lasso_tol),
            seed=int(seed) + i,
        )
        protein_coef = coef[: len(proteins)]
        coef_blocks.append(pd.Series(protein_coef, index=proteins, name=f"subsample_{i}"))
    coef_df = pd.concat(coef_blocks, axis=1)
    selected = coef_df.abs() > 1e-12
    ranking = pd.DataFrame(
        {
            "feature": coef_df.index,
            "display_name": [feature_display_name(feature) for feature in coef_df.index],
            "selection_frequency": selected.mean(axis=1).to_numpy(dtype=float),
            "mean_abs_coef": coef_df.abs().mean(axis=1).to_numpy(dtype=float),
            "mean_signed_coef": coef_df.mean(axis=1).to_numpy(dtype=float),
            "positive_frequency": (coef_df > 1e-12).mean(axis=1).to_numpy(dtype=float),
            "negative_frequency": (coef_df < -1e-12).mean(axis=1).to_numpy(dtype=float),
        }
    )
    ranking = ranking.sort_values(
        ["selection_frequency", "mean_abs_coef", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "stability_rank", np.arange(1, len(ranking) + 1))
    qc.update(
        {
            "training_rows_for_window": int(len(y)),
            "training_cases_for_window": int(y.sum()),
            "training_controls_for_window": int(len(y) - int(y.sum())),
            "subsamples_completed": int(len(coef_blocks)),
            "adjustment_covariates": ";".join(adjustment_features),
        }
    )
    return ranking, qc


def integrate_window_rankings(rankings: dict[str, pd.DataFrame], windows: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for window in windows:
        sub = rankings[window].copy()
        sub = sub[
            [
                "feature",
                "display_name",
                "stability_rank",
                "selection_frequency",
                "mean_abs_coef",
                "mean_signed_coef",
            ]
        ]
        sub["window"] = window
        frames.append(sub)
    long_df = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, object]] = []
    for feature, sub in long_df.groupby("feature", sort=False):
        record: dict[str, object] = {
            "feature": feature,
            "display_name": feature_display_name(str(feature)),
            "integrated_lasso_score": float(pd.to_numeric(sub["mean_abs_coef"], errors="coerce").fillna(0).sum()),
            "window_count_frequency_ge80": int((pd.to_numeric(sub["selection_frequency"], errors="coerce").fillna(0) >= 0.80).sum()),
            "mean_selection_frequency": float(pd.to_numeric(sub["selection_frequency"], errors="coerce").fillna(0).mean()),
            "best_stability_rank": int(pd.to_numeric(sub["stability_rank"], errors="coerce").min()),
            "contains_nectin2": str(feature).lower() == "nectin2",
        }
        for _, row in sub.iterrows():
            window = str(row["window"])
            record[f"rank_{window}"] = int(row["stability_rank"])
            record[f"selection_frequency_{window}"] = float(row["selection_frequency"])
            record[f"mean_abs_coef_{window}"] = float(row["mean_abs_coef"])
            record[f"mean_signed_coef_{window}"] = float(row["mean_signed_coef"])
        rows.append(record)
    out = pd.DataFrame(rows)
    out = out.sort_values(
        [
            "integrated_lasso_score",
            "window_count_frequency_ge80",
            "mean_selection_frequency",
            "best_stability_rank",
            "feature",
        ],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)
    out.insert(0, "integrated_rank", np.arange(1, len(out) + 1))
    return out


def compress_candidate_pool(
    train_df: pd.DataFrame,
    candidate_pool: list[str],
    adjustment_features: list[str],
    windows: list[str],
    *,
    control_mode: str,
    config: SelectionConfig,
    panel_size: int,
    seed: int,
) -> tuple[list[str], pd.DataFrame]:
    """Use a second training-only L1 model to reduce the candidate pool."""
    if panel_size < 1 or panel_size > len(candidate_pool):
        raise ValueError("panel_size must not exceed the candidate-pool size.")
    model_features = list(dict.fromkeys(candidate_pool + adjustment_features))
    blocks: list[pd.DataFrame] = []
    for index, window in enumerate(windows):
        window_df, y, _ = build_exact_window(train_df, window, control_mode=control_mode)
        coef = _fit_sparse_lasso(
            window_df[model_features],
            y,
            c=float(config.lasso_c),
            class_weight=config.class_weight,
            max_iter=int(config.lasso_max_iter),
            tol=float(config.lasso_tol),
            seed=int(seed) + index,
        )[: len(candidate_pool)]
        blocks.append(
            pd.DataFrame(
                {
                    "feature": candidate_pool,
                    "window": window,
                    "abs_coefficient": np.abs(coef),
                    "signed_coefficient": coef,
                }
            )
        )
    coefficients = pd.concat(blocks, ignore_index=True)
    summary = (
        coefficients.groupby("feature", as_index=False)
        .agg(
            integrated_abs_coefficient=("abs_coefficient", "sum"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            mean_signed_coefficient=("signed_coefficient", "mean"),
            nonzero_windows=("abs_coefficient", lambda values: int((values > 1e-12).sum())),
        )
        .sort_values(
            ["integrated_abs_coefficient", "nonzero_windows", "feature"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    nonzero = summary[summary["integrated_abs_coefficient"].gt(1e-12)]
    if len(nonzero) < int(panel_size):
        raise RuntimeError(
            f"Secondary L1 retained only {len(nonzero)} non-zero proteins; "
            f"{panel_size} are required."
        )
    selected = nonzero.head(panel_size)["feature"].astype(str).tolist()
    summary.insert(0, "compression_rank", np.arange(1, len(summary) + 1))
    summary["selected_in_panel"] = summary["feature"].isin(selected)
    return selected, summary


def _fit_lasso_predict(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    *,
    c: float,
    class_weight: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_train = coerce_model_frame(x_train)
    x_test = coerce_model_frame(x_test)
    empty_columns = x_train.columns[x_train.isna().all()]
    if len(empty_columns):
        x_train.loc[:, empty_columns] = 0.0
        x_test.loc[:, empty_columns] = 0.0
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    C=float(c),
                    class_weight=_class_weight_arg(class_weight),
                    max_iter=4000,
                    random_state=int(seed),
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        model.fit(x_train, y_train.to_numpy(dtype=int))
    predictions = model.predict_proba(x_test)[:, 1]
    coefficients = model.named_steps["clf"].coef_.ravel()
    return predictions, coefficients


def fit_models_for_window(
    train_window_df: pd.DataFrame,
    train_y: pd.Series,
    test_window_df: pd.DataFrame,
    selected_features: list[str],
    demographic_features: list[str],
    *,
    logistic_c: float,
    class_weight: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    predictions: dict[str, np.ndarray] = {}
    x_train_protein = train_window_df[selected_features].reset_index(drop=True)
    x_test_protein = test_window_df[selected_features].reset_index(drop=True)
    predictions["protein"], protein_coefficients = _fit_lasso_predict(
        x_train_protein,
        train_y.reset_index(drop=True),
        x_test_protein,
        c=logistic_c,
        class_weight=class_weight,
        seed=seed + 1,
    )
    if demographic_features:
        demo_features = [feature for feature in demographic_features if feature in train_window_df.columns]
        x_train_demo = train_window_df[demo_features].reset_index(drop=True)
        x_test_demo = test_window_df[demo_features].reset_index(drop=True)
        predictions["demographic"], _ = _fit_lasso_predict(
            x_train_demo,
            train_y.reset_index(drop=True),
            x_test_demo,
            c=logistic_c,
            class_weight=class_weight,
            seed=seed + 2,
        )
        combined_features = list(dict.fromkeys(selected_features + demo_features))
        predictions["combined"], _ = _fit_lasso_predict(
            train_window_df[combined_features].reset_index(drop=True),
            train_y.reset_index(drop=True),
            test_window_df[combined_features].reset_index(drop=True),
            c=logistic_c,
            class_weight=class_weight,
            seed=seed + 3,
        )
    coefficient_table = pd.DataFrame(
        {
            "feature": selected_features,
            "display_name": [feature_display_name(feature) for feature in selected_features],
            "standardized_coefficient": protein_coefficients,
        }
    )
    return predictions, coefficient_table


def calibration_intercept_slope(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    clipped = np.clip(pred, 1e-6, 1 - 1e-6)
    logit_pred = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            warnings.simplefilter("ignore", category=UserWarning)
            model = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
            model.fit(logit_pred, y)
    except Exception:
        return np.nan, np.nan
    return float(model.intercept_[0]), float(model.coef_[0][0])


def metric_values(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    intercept, slope = calibration_intercept_slope(y, pred)
    return {
        "auc": float(roc_auc_score(y, pred)),
        "average_precision": float(average_precision_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def bootstrap_interval(
    y: np.ndarray,
    pred: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    if int(n_bootstrap) <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(int(seed))
    values: list[float] = []
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=float)
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[idx])) < 2 and metric_fn is not brier_score_loss:
            continue
        try:
            values.append(float(metric_fn(y[idx], pred[idx])))
        except Exception:
            continue
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize_predictions(
    pred_df: pd.DataFrame,
    split_metrics: pd.DataFrame,
    *,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for panel_id, panel_sub in pred_df.groupby("panel_id", sort=False):
        for window, window_sub in panel_sub.groupby("window", sort=False):
            for model_key, pred_col in [
                ("protein", "p_protein"),
                ("demographic", "p_demographic"),
                ("combined", "p_combined"),
            ]:
                if pred_col not in window_sub.columns:
                    continue
                y = window_sub["target_y"].to_numpy(dtype=int)
                pred = window_sub[pred_col].to_numpy(dtype=float)
                if len(np.unique(y)) < 2:
                    continue
                row: dict[str, object] = {
                    "panel_id": panel_id,
                    "panel_size": int(window_sub["panel_size"].iloc[0]),
                    "candidate_pool_size": int(window_sub["candidate_pool_size"].iloc[0]),
                    "window": window,
                    "window_label": window_sub["window_label"].iloc[0],
                    "model_key": model_key,
                    "model": MODEL_LABELS[model_key],
                    "n": int(len(y)),
                    "cases": int(y.sum()),
                    "controls": int(len(y) - int(y.sum())),
                    "selection_rule": "outer_train_stability_pool_secondary_l1",
                    "final_model": "l1_logistic_regression",
                }
                row.update(metric_values(y, pred))
                row["auc_ci_low"], row["auc_ci_high"] = bootstrap_interval(
                    y,
                    pred,
                    roc_auc_score,
                    n_bootstrap=bootstrap,
                    seed=seed + len(rows),
                )
                row["average_precision_ci_low"], row["average_precision_ci_high"] = bootstrap_interval(
                    y,
                    pred,
                    average_precision_score,
                    n_bootstrap=bootstrap,
                    seed=seed + 500 + len(rows),
                )
                row["brier_ci_low"], row["brier_ci_high"] = bootstrap_interval(
                    y,
                    pred,
                    brier_score_loss,
                    n_bootstrap=bootstrap,
                    seed=seed + 1000 + len(rows),
                )
                sub_metrics = split_metrics[
                    split_metrics["panel_id"].eq(panel_id)
                    & split_metrics["window"].eq(window)
                    & split_metrics["model_key"].eq(model_key)
                ]
                row["mean_fold_auc"] = float(sub_metrics["auc"].mean()) if not sub_metrics.empty else np.nan
                row["sd_fold_auc"] = float(sub_metrics["auc"].std(ddof=0)) if not sub_metrics.empty else np.nan
                row["mean_fold_calibration_slope"] = (
                    float(sub_metrics["calibration_slope"].mean()) if not sub_metrics.empty else np.nan
                )
                rows.append(row)
    return pd.DataFrame(rows)


def feature_frequency(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for panel_id, sub in panel_df.groupby("panel_id", sort=False):
        n_folds = max(int(sub["outer_fold"].nunique()), 1)
        counts: dict[str, int] = {}
        for features in sub["selected_features"].fillna(""):
            for feature in str(features).split(";"):
                if feature:
                    counts[feature] = counts.get(feature, 0) + 1
        for feature, count in counts.items():
            rows.append(
                {
                    "panel_id": panel_id,
                    "panel_size": int(sub["panel_size"].iloc[0]),
                    "candidate_pool_size": int(sub["candidate_pool_size"].iloc[0]),
                    "feature": feature,
                    "display_name": feature_display_name(feature),
                    "outer_fold_selection_count": count,
                    "outer_fold_selection_frequency": count / n_folds,
                    "is_nectin2": feature == "nectin2",
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "panel_id",
                "panel_size",
                "candidate_pool_size",
                "feature",
                "display_name",
                "outer_fold_selection_count",
                "outer_fold_selection_frequency",
                "is_nectin2",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["panel_id", "outer_fold_selection_count", "feature"],
        ascending=[True, False, True],
    )
