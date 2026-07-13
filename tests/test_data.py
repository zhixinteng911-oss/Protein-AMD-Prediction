from __future__ import annotations

import pandas as pd
import pytest

from ML.lasso_workflow.data import build_exact_window, read_analysis_frame
from conftest import make_synthetic_cohort


def test_exact_windows_use_event_free_controls() -> None:
    frame = pd.DataFrame(
        {
            "target_y": [0, 0, 1, 1, 1],
            "BL2Target_yrs": [4.0, 12.0, 2.0, 7.0, 11.0],
        }
    )
    near, y_near, _ = build_exact_window(frame, "exact_0_3", control_mode="event_free")
    mid, y_mid, _ = build_exact_window(frame, "exact_3_9", control_mode="event_free")
    assert len(near) == 3 and int(y_near.sum()) == 1
    assert len(mid) == 2 and int(y_mid.sum()) == 1


def test_loader_requires_stability_adjustment_covariates(tmp_path) -> None:
    frame = make_synthetic_cohort().drop(columns="PC20")
    input_path = tmp_path / "cohort.csv"
    manifest_path = tmp_path / "proteins.csv"
    frame.to_csv(input_path, index=False)
    pd.DataFrame({"protein": ["nectin2", "gdf15"]}).to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="genetic_pc20"):
        read_analysis_frame(input_path, proteins_file=manifest_path, id_col="Participant.ID")


def test_loader_rejects_duplicate_participants(tmp_path) -> None:
    frame = make_synthetic_cohort()
    frame.loc[1, "Participant.ID"] = frame.loc[0, "Participant.ID"]
    input_path = tmp_path / "cohort.csv"
    manifest_path = tmp_path / "proteins.csv"
    frame.to_csv(input_path, index=False)
    pd.DataFrame({"protein": ["nectin2", "gdf15"]}).to_csv(manifest_path, index=False)
    with pytest.raises(ValueError, match="must be unique"):
        read_analysis_frame(input_path, proteins_file=manifest_path, id_col="Participant.ID")


def test_loader_excludes_missing_outcome_or_followup(tmp_path) -> None:
    frame = make_synthetic_cohort()
    frame.loc[0, "target_y"] = None
    frame.loc[1, "BL2Target_yrs"] = None
    input_path = tmp_path / "cohort.csv"
    manifest_path = tmp_path / "proteins.csv"
    frame.to_csv(input_path, index=False)
    pd.DataFrame({"protein": ["nectin2", "gdf15"]}).to_csv(manifest_path, index=False)
    loaded, *_ = read_analysis_frame(
        input_path,
        proteins_file=manifest_path,
        id_col="Participant.ID",
    )
    assert len(loaded) == len(frame) - 2
