from __future__ import annotations

from ML.lasso_workflow.modeling import (
    SelectionConfig,
    compress_candidate_pool,
    run_window_stability_train_only,
)
from conftest import make_synthetic_cohort


def test_stability_and_secondary_l1_use_declared_protein_pool() -> None:
    frame = make_synthetic_cohort().rename(
        columns={"Age_at_recruitment": "age", "sex_binary": "sex", "amd_prs": "prs"}
    )
    frame = frame.rename(columns={f"PC{i}": f"pc{i}" for i in range(1, 21)})
    frame["noise2"] = float("nan")
    proteins = ["nectin2", "gdf15", "nefl", "mmp12", "noise1", "noise2"]
    adjustment = ["age", "sex", "prs"] + [f"pc{i}" for i in range(1, 21)]
    config = SelectionConfig(2, 0.7, 10.0, 0.1, 1000, 1e-3, "balanced", 2022)
    ranking, _ = run_window_stability_train_only(
        frame,
        proteins,
        adjustment,
        "exact_0_3",
        control_mode="event_free",
        config=config,
        seed=2022,
    )
    assert set(ranking["feature"]) == set(proteins)
    assert ranking.loc[ranking["feature"].eq("noise2"), "mean_abs_coef"].eq(0).all()
    selected, compression = compress_candidate_pool(
        frame,
        proteins[:4],
        adjustment,
        ["exact_0_3", "exact_3_9", "exact_gt9"],
        control_mode="event_free",
        config=config,
        panel_size=2,
        seed=2022,
    )
    assert len(selected) == 2
    assert set(selected).issubset(set(proteins[:4]))
    assert int(compression["selected_in_panel"].sum()) == 2
