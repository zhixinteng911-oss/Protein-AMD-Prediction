#!/usr/bin/env python3
"""Run the Cox workflow in order."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-file", required=True, help="Broad H35 cohort CSV.")
    parser.add_argument("--prot-file", required=True, help="Proteomics CSV.")
    parser.add_argument("--prs-file", required=True, help="PRS CSV.")
    parser.add_argument("--protein-list-file", required=True, help="Protein list CSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory root.")
    parser.add_argument(
        "--censor-date",
        required=True,
        help="Administrative censoring date in YYYY-MM-DD format.",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=25)
    return parser.parse_args()


def run_step(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    h353_file = out_dir / "amd_cox_analysis_ready_with_alcohol_h353_only.csv"
    analysis_file = out_dir / "amd_final_analysis_dataset_h353_only.csv"

    run_step(
        [
            sys.executable,
            str(Path(__file__).with_name("01_filter_h353.py")),
            "--input-file",
            str(Path(args.broad_file).expanduser().resolve()),
            "--output-file",
            str(h353_file),
            "--log-file",
            str(out_dir / "h353_filtering_summary.json"),
        ]
    )
    run_step(
        [
            sys.executable,
            str(Path(__file__).with_name("02_build_analysis_dataset.py")),
            "--prot-file",
            str(Path(args.prot_file).expanduser().resolve()),
            "--cov-file",
            str(h353_file),
            "--prs-file",
            str(Path(args.prs_file).expanduser().resolve()),
            "--censor-date",
            args.censor_date,
            "--out-file",
            str(analysis_file),
        ]
    )
    run_step(
        [
            sys.executable,
            str(Path(__file__).with_name("03_run_cox.py")),
            "--data-file",
            str(analysis_file),
            "--protein-list-file",
            str(Path(args.protein_list_file).expanduser().resolve()),
            "--out-dir",
            str(out_dir / "cox_h353"),
            "--n-jobs",
            str(args.n_jobs),
            "--batch-size",
            str(args.batch_size),
        ]
    )


if __name__ == "__main__":
    main()
