# H35.3 Cox association workflow

This directory constructs the H35.3-restricted incident AMD cohort and runs
protein-wise Cox proportional hazards models.

## Analysis sequence

1. `01_filter_h353.py` defines incident H35.3 events from UK Biobank field
   `131182`, applies death/loss/administrative censoring, and removes
   non-positive follow-up.
2. `02_build_analysis_dataset.py` merges proteomics, covariates, and AMD PRS,
   then creates the prespecified transformed covariates.
3. `03_run_cox.py` fits unadjusted, age/sex-adjusted, and fully adjusted models
   for each protein and applies Benjamini-Hochberg FDR correction separately
   within each model.

All three models for a protein use the same protein-specific complete-case
sample, so changes in the hazard ratio across adjustment levels are not caused
by changes in participants. Protein manifests may be either a one-protein-per-
row CSV with a `protein` header or a wide CSV whose header contains protein
names.

The primary association analysis is unpenalized (`--penalizer 0`). A positive
L2 penalizer, such as `--penalizer 0.01`, is available only as a sensitivity
analysis for convergence or separation; penalized estimates should not replace
the primary inferential results without an explicit methods change.

## Example

Run heavy analyses on AgentServer from a project directory under
`/data/users/tzx/projects/`:

```bash
python cox/03_run_cox.py \
  --data-file /data/users/tzx/data/ukb/raw/amd_final_analysis_dataset_h353_only.csv \
  --protein-list-file ML/candidate_proteins.csv \
  --out-dir outputs/cox_h353_primary \
  --n-jobs 10 \
  --penalizer 0
```

The output includes raw estimates, model-wise multiplicity-adjusted results,
fit warnings, exclusions, per-batch checkpoints, and a JSON provenance record.
