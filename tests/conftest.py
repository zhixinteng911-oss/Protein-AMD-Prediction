from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_cohort(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.repeat(["control", "near", "mid", "long"], [60, 12, 12, 12])
    n = len(groups)
    event = (groups != "control").astype(int)
    years = np.empty(n)
    years[groups == "control"] = rng.uniform(10, 16, size=(groups == "control").sum())
    years[groups == "near"] = rng.uniform(0.2, 3, size=(groups == "near").sum())
    years[groups == "mid"] = rng.uniform(3.2, 9, size=(groups == "mid").sum())
    years[groups == "long"] = rng.uniform(9.2, 15, size=(groups == "long").sum())
    frame = pd.DataFrame(
        {
            "Participant.ID": [f"p{i:04d}" for i in range(n)],
            "target_y": event,
            "BL2Target_yrs": years,
            "Age_at_recruitment": rng.normal(60, 5, n) + event * 2,
            "sex_binary": rng.integers(0, 2, n),
            "bmi_log_z": rng.normal(0, 1, n),
            "smoker_current": rng.integers(0, 2, n),
            "ldlr": rng.normal(0, 1, n),
            "alcohol_frequent": rng.integers(0, 2, n),
            "amd_prs": rng.normal(0, 1, n) + event * 0.3,
            "nectin2": rng.normal(0, 1, n) + event,
            "gdf15": rng.normal(0, 1, n) + event * 0.7,
            "nefl": rng.normal(0, 1, n) + event * 0.5,
            "mmp12": rng.normal(0, 1, n) + event * 0.4,
            "noise1": rng.normal(0, 1, n),
            "noise2": rng.normal(0, 1, n),
        }
    )
    for index in range(1, 21):
        frame[f"PC{index}"] = rng.normal(0, 1, n)
    return frame
