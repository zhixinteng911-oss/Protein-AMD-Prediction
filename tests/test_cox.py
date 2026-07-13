from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_cox", ROOT / "cox" / "03_run_cox.py")
assert SPEC is not None and SPEC.loader is not None
RUN_COX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_COX)


def test_read_protein_list_accepts_long_and_wide_manifests(tmp_path: Path) -> None:
    long_path = tmp_path / "long.csv"
    long_path.write_text("protein\nNECTIN2\nGDF15\n", encoding="utf-8")
    wide_path = tmp_path / "wide.csv"
    wide_path.write_text("NECTIN2,GDF15\n", encoding="utf-8")

    assert RUN_COX.read_protein_list(long_path) == ["NECTIN2", "GDF15"]
    assert RUN_COX.read_protein_list(wide_path) == ["NECTIN2", "GDF15"]


def test_read_protein_list_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.csv"
    path.write_text("protein\nNECTIN2\nNECTIN2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates"):
        RUN_COX.read_protein_list(path)


def test_models_share_the_fully_adjusted_complete_case_sample(monkeypatch) -> None:
    n = 80
    frame = pd.DataFrame(
        {
            RUN_COX.TIME_COL: np.linspace(1, 12, n),
            RUN_COX.EVENT_COL: np.tile([0, 0, 0, 1], n // 4),
            "protein_a": np.linspace(-2, 2, n),
            "age_z": np.linspace(-1, 1, n),
            "sex_binary": np.tile([0, 1], n // 2),
            "full_covariate": np.linspace(0, 1, n),
        }
    )
    frame.loc[:9, "full_covariate"] = np.nan
    fitted_sizes: list[int] = []

    def fake_fit(df_fit, protein, penalizer):
        fitted_sizes.append(len(df_fit))
        return {
            "hr": 1.0,
            "hr_lower_ci": 0.9,
            "hr_upper_ci": 1.1,
            "p_value": 0.5,
        }, []

    monkeypatch.setattr(RUN_COX, "fit_cox", fake_fit)
    results, warnings, exclusions = RUN_COX.analyse_protein(
        "protein_a",
        frame,
        ["age_z", "sex_binary"],
        ["age_z", "sex_binary", "full_covariate"],
        penalizer=0.0,
    )

    assert not warnings
    assert not exclusions
    assert fitted_sizes == [70, 70, 70]
    assert {row["n_samples"] for row in results} == {70}


def test_multiple_testing_is_applied_within_each_model() -> None:
    results = pd.DataFrame(
        {
            "protein": ["a", "b", "a", "b"],
            "model": ["m1", "m1", "m2", "m2"],
            "p_value": [0.01, 0.04, 0.2, 0.8],
        }
    )

    adjusted = RUN_COX.apply_multiple_testing(results)
    by_model = adjusted.set_index(["model", "protein"])
    assert by_model.loc[("m1", "a"), "fdr_p"] == pytest.approx(0.02)
    assert by_model.loc[("m2", "a"), "fdr_p"] == pytest.approx(0.4)
