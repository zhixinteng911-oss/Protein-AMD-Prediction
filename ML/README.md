# AMD ML Mainline

This `ML/` tree contains one publication-facing pipeline:

- `elastic-net rerank`
- `full-data 5-fold LightGBM SFS`
- `5-fold OOF ROC`
- `fold-aggregated SHAP`

Core files:

- `shared.py`
- `within_3yr/01_rerank.py`
- `within_3yr/02_sfs.py`
- `within_3yr/03_plot.py`
- `within_3yr/04_roc.py`
- `within_3yr/05_shap.py`
