#!/usr/bin/env python3
from __future__ import annotations

import argparse

from lasso_workflow.workflow import run_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Five-fold LASSO-stability validation for AMD temporal protein prediction."
        )
    )
    parser.add_argument(
        "--input", "--input-file", dest="input",
        default="/data/users/tzx/data/ukb/raw/amd_final_analysis_dataset_h353_only.csv",
        help="Analysis-ready UK Biobank/Olink CSV.",
    )
    parser.add_argument(
        "--proteins",
        default="/data/users/tzx/projects/nectin2-amd/Protein-AMD-Prediction/ML/candidate_proteins.csv",
        help="Candidate protein manifest CSV.",
    )
    parser.add_argument(
        "--out", "--out-dir", dest="out",
        default="/data/users/tzx/projects/nectin2-amd/outputs/nested5fold_lasso1000_selected_panel",
        help="Output directory.",
    )
    parser.add_argument("--id-col", default="Participant.ID", help="Unique participant identifier column.")
    parser.add_argument(
        "--windows",
        nargs="+",
        default=["exact_0_3", "exact_3_9", "exact_gt9"],
        help="Temporal windows to evaluate.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["event_free", "horizon"],
        default="event_free",
        help="Control definition for each temporal endpoint.",
    )
    parser.add_argument("--candidate-pool-size", type=int, default=20)
    parser.add_argument("--panel-size", type=int, default=10)
    parser.add_argument("--outer-folds", type=int, default=5, help="Participant-level outer folds.")
    parser.add_argument("--seed", type=int, default=2022, help="Random seed.")
    parser.add_argument("--n-subsamples", "--subsamples", dest="n_subsamples", type=int, default=1000, help="LASSO stability subsamples per outer fold and window.")
    parser.add_argument("--sample-frac", type=float, default=0.70, help="Case sampling fraction for stability selection.")
    parser.add_argument("--control-case-ratio", type=float, default=20.0, help="Controls sampled per case in stability selection.")
    parser.add_argument("--lasso-c", type=float, default=0.1, help="L1 logistic inverse regularization for stability selection.")
    parser.add_argument("--lasso-max-iter", type=int, default=4000)
    parser.add_argument("--lasso-tol", type=float, default=1e-3)
    parser.add_argument("--logistic-c", type=float, default=1.0, help="Final L1-logistic inverse regularization.")
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help=(
            "Class weighting for both LASSO stability and final L1-logistic models."
        ),
    )
    parser.add_argument("--bootstrap", type=int, default=500, help="Bootstrap replicates for metric intervals.")
    return parser.parse_args()


def main() -> None:
    run_workflow(parse_args())


if __name__ == "__main__":
    main()
