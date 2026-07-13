WINDOWS = [
    ("exact_0_3", "0-3 years"),
    ("exact_3_9", "3-9 years"),
    ("exact_gt9", ">9 years"),
]

WINDOW_SPECS = {
    "exact_0_3": {
        "label": "0-3 years",
        "case_min": 0.0,
        "case_max": 3.0,
        "control_followup_min": 3.0,
    },
    "exact_3_9": {
        "label": "3-9 years",
        "case_min": 3.0,
        "case_max": 9.0,
        "control_followup_min": 9.0,
    },
    "exact_gt9": {
        "label": ">9 years",
        "case_min": 9.0,
        "case_max": None,
        "control_followup_min": 9.0,
    },
}

OUTCOME_ALIASES = {
    "target_y": ["target_y", "incident_amd", "incident AMD"],
    "BL2Target_yrs": ["BL2Target_yrs", "years_to_amd", "years to AMD"],
}

ID_ALIASES = ["Participant.ID", "Participant_ID", "eid", "id", "participant_id"]

DEMO_COLS = [
    "Age_at_recruitment",
    "sex_binary",
    "bmi_log_z",
    "smoker_current",
    "ldlr",
    "alcohol_frequent",
]

DEMO_ALIASES = {
    "Age_at_recruitment": [
        "Age_at_recruitment",
        "Age at recruitment",
        "age_at_recruitment",
        "age",
        "age_z",
        "f_21022",
        "f_21022_0_0",
    ],
    "sex_binary": ["sex_binary", "sex", "f_31", "f_31_0_0"],
    "bmi_log_z": [
        "bmi_log_z",
        "bmi",
        "body_mass_index_bmi",
        "f_21001",
        "f_21001_0",
        "f_21001_0_0",
    ],
    "smoker_current": ["smoker_current", "current_smoking"],
    "ldlr": ["ldlr"],
    "alcohol_frequent": ["alcohol_frequent", "alcohol"],
}

ADJUSTMENT_ALIASES = {
    "Age_at_recruitment": DEMO_ALIASES["Age_at_recruitment"],
    "sex_binary": DEMO_ALIASES["sex_binary"],
    "amd_prs": ["prs_amd_z", "amd_prs_z", "prs_amd", "amd_prs", "prs"],
    **{
        f"genetic_pc{i}": [
            f"genetic_pc{i}_filled",
            f"Genetic principal components | Array {i}",
            f"PC{i}",
        ]
        for i in range(1, 21)
    },
}

MODEL_LABELS = {
    "protein": "Protein",
    "demographic": "Demographic",
    "combined": "Protein + demographic",
}
