# AMD H35.3 Cox Mainline

This directory contains the maintained Cox workflow for the H35.3-restricted AMD analysis.

Maintained flow:

1. `01_filter_h353.py`
   - restrict the broad H35 cohort to an H35.3-focused cohort definition
2. `02_build_analysis_dataset.py`
   - merge the H35.3 cohort with proteomics, PRS, and processed covariates
3. `03_run_cox.py`
   - run protein-wise Cox models for:
     - Model 1: unadjusted
     - Model 2: age + sex
     - Model 3: fully adjusted

Current design choices:

- the H35.3 cohort is the anchor table for the merged analysis dataset
- continuous and categorical covariates are transformed without pre-imputing away missingness
- complete-case filtering is applied separately for each Cox model, rather than once across all models
- batch outputs, warnings, exclusions, and metadata are written to the chosen output directory
