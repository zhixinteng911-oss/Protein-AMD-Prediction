# Within_3yr Mainline

This directory is the canonical AMD protein-prediction pipeline for incident AMD within 3 years.

Maintained flow:

1. `01_rerank.py`
   - elastic-net pre-ranking on the full prediction cohort
2. `02_sfs.py`
   - full-data 5-fold LightGBM sequential forward selection
3. `03_plot.py`
   - SFS figure using the selected panel as the default display
4. `04_roc.py`
   - full-data 5-fold out-of-fold ROC evaluation
5. `05_shap.py`
   - fold-aggregated SHAP summary and sample-level SHAP detail for the final protein panel

Key design choices:

- one cohort definition is used across reranking, SFS, ROC, and SHAP
- future cases beyond the 3-year horizon are retained as non-events for the 3-year prediction task
- final ROC and SHAP outputs are based on full-data 5-fold out-of-fold evaluation rather than a separate holdout split

Files outside this flow are legacy or exploratory and should not be mixed into the final release.
