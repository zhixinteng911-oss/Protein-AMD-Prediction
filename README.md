# Protein-AMD-Prediction

Code for the AMD proteomics analyses in the manuscript.

## Cox

- `01_filter_h353.py`: keep the H35.3 AMD outcome
- `02_build_analysis_dataset.py`: build the final analysis table
- `03_run_cox.py`: run the Cox models and FDR correction
- `run_all.py`: run the Cox workflow from start to finish

## ML

Prediction windows:

- `within_3yr`
- `from_3_to_9yr`
- `after_9yr`
- `full_cohort`

Each window has the same five steps:

- `01_rerank.py`
- `02_sfs.py`
- `03_plot.py`
- `04_roc.py`
- `05_shap.py`
