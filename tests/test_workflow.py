from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from ML.lasso_workflow.workflow import run_workflow
from conftest import make_synthetic_cohort


def test_workflow_writes_candidate_pool_panel_and_heldout_predictions(tmp_path) -> None:
    input_path = tmp_path / "cohort.csv"
    manifest_path = tmp_path / "proteins.csv"
    out_dir = tmp_path / "results"
    make_synthetic_cohort().to_csv(input_path, index=False)
    proteins = ["nectin2", "gdf15", "nefl", "mmp12", "noise1", "noise2"]
    pd.DataFrame({"protein": proteins}).to_csv(manifest_path, index=False)
    args = SimpleNamespace(
        input=str(input_path),
        proteins=str(manifest_path),
        out=str(out_dir),
        id_col="Participant.ID",
        windows=["exact_0_3", "exact_3_9", "exact_gt9"],
        control_mode="event_free",
        candidate_pool_size=4,
        panel_size=2,
        outer_folds=2,
        seed=2022,
        n_subsamples=2,
        sample_frac=0.7,
        control_case_ratio=10.0,
        lasso_c=0.1,
        lasso_max_iter=1000,
        lasso_tol=1e-3,
        logistic_c=1.0,
        class_weight="balanced",
        bootstrap=5,
    )
    run_workflow(args)
    expected = {
        "outer_predictions.csv",
        "pooled_metrics.csv",
        "outer_fold_candidate_pools.csv",
        "outer_fold_panel_compression.csv",
        "outer_fold_selected_panels.csv",
        "fig4a_top20_integrated_ranking.csv",
        "fig4b_selected_panel_outer_heldout_roc.pdf",
        "fig4c_consensus_coefficients.csv",
        "fig4c_consensus_coefficients.pdf",
        "run_config.json",
        "README.md",
    }
    assert expected.issubset({path.name for path in out_dir.iterdir()})
    predictions = pd.read_csv(out_dir / "outer_predictions.csv")
    assert predictions["panel_size"].eq(2).all()
    assert not predictions.duplicated(["window", "participant_row"]).any()
    assert predictions.filter(like="p_").apply(lambda col: col.between(0, 1).all()).all()
    assert predictions.groupby("window")["outer_fold"].nunique().eq(2).all()
    consensus = pd.read_csv(out_dir / "fig4c_consensus_coefficients.csv")
    assert len(consensus) == 2
