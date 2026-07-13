# Machine-learning analysis

## Workflow

Participants are divided into five stratified folds. In each iteration, four folds are used for training and the remaining fold is reserved for evaluation.
The candidate manifest contains the 92 proteins identified by the preceding Cox analysis.

Within the four training folds:

1. Three endpoints are constructed for 0-3, 3-9 and >9 years before AMD diagnosis.
2. Each endpoint undergoes 1,000 repeated 70% subsampling LASSO analyses. Age, sex, AMD polygenic risk score and genetic principal components 1-20 are included as adjustment covariates.
3. Protein coefficients are integrated across the three windows. The 20 highest-ranked proteins form the candidate pool.
4. A second training-only L1 model reduces the candidate pool to a fold-specific 10-protein panel.
5. Protein-only, demographic-only and combined L1-logistic models are fitted in the training data and applied to the held-out fold.

Imputation, scaling, feature selection and model fitting are restricted to the training folds. Predictions from the five held-out folds are pooled for ROC/AUC estimation.

## Run

Run on the authorized server from `/data/users/tzx/projects/nectin2-amd/Protein-AMD-Prediction/`:

```bash
python3 ML/run_lasso_fig4.py \
  --input /data/users/tzx/data/ukb/raw/amd_final_analysis_dataset_h353_only.csv \
  --proteins ML/candidate_proteins.csv \
  --out /data/users/tzx/projects/nectin2-amd/outputs/nested5fold_lasso1000_selected_panel \
  --candidate-pool-size 20 \
  --panel-size 10 \
  --outer-folds 5 \
  --n-subsamples 1000 \
  --sample-frac 0.70 \
  --control-case-ratio 20 \
  --lasso-c 0.1 \
  --logistic-c 1.0 \
  --class-weight balanced \
  --bootstrap 500 \
  --seed 2022
```

## Main outputs

- `outer_fold_selected_panels.csv`: fold-specific 10-protein panels.
- `outer_fold_window_protein_coefficients.csv`: fold- and window-specific standardized coefficients.
- `outer_predictions.csv`: held-out predictions.
- `pooled_metrics.csv`: pooled AUCs and confidence intervals.
- `fig4a_top20_integrated_ranking.csv`: fold-averaged integrated stability ranking.
- `fig4b_selected_panel_outer_heldout_roc.*`: ROC figure.
- `fig4c_consensus_coefficients.*`: post-cross-validation consensus coefficient profile.
- `window_case_sparsity_qc.csv` and `run_config.json`: run-level quality control.
