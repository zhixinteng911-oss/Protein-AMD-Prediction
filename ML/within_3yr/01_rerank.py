#!/usr/bin/env python3
"""Elastic-net reranking for the canonical within_3yr prediction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import (
    get_output_dir,
    make_folds,
    prepare_prediction_payload,
    rank_features_elasticnet_cv,
)

WINDOW = "within_3yr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Elastic-net rerank for the within_3yr prediction pipeline.")
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--base-out", default="outputs")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    args = parser.parse_args()

    payload = prepare_prediction_payload(args.input_file, WINDOW)
    out_dir = get_output_dir(args.base_out, WINDOW)
    x_protein = payload["X_protein"]
    y = payload["y"]
    folds = make_folds(y, strict=True, label="elasticnet feature rerank")

    rank_df = rank_features_elasticnet_cv(
        x_protein,
        y,
        folds,
        random_state=args.seed,
        c=args.c,
        l1_ratio=args.l1_ratio,
    )
    rank_df.to_csv(out_dir / "s03_reranked.csv", index=False)
    print(rank_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
