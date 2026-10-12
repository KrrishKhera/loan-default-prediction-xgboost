# Exploratory SHAP explanations

## Scope and reproducibility

`loan_default.explain` loads `models/selected_model.joblib` read-only and explains its existing fitted XGBoost pipeline. It does not fit or replace the model. It checks the artifact's ordered 11-feature schema against the project allowlist and excludes `loan_status` from inputs.

For the default seed (42), the module uses a deterministic 80-row background and a deterministic 300-row explanation sample drawn only from the development training partition returned by `split_data`. It never scores or passes the previously evaluated test partition to SHAP. Individual examples are a deterministic subset of those training rows. Output files are written to `reports/exploratory_shap/` and explicitly describe themselves as exploratory explanations of the existing model.

Run from the repository root:

```bash
PYTHONPATH=src python -m loan_default.explain \
  --csv data/raw/credit_risk_dataset.csv \
  --artifact models/selected_model.joblib \
  --output-dir reports/exploratory_shap \
  --random-state 42 \
  --background-size 80 \
  --explanation-size 300 \
  --individual-count 3
```

The command creates:

- `global_feature_importance.csv`: each raw feature's mean absolute and mean signed SHAP contribution on the default-probability scale. One-hot columns are summed back to their source categorical feature before these summaries are computed.
- `global_feature_importance.png`: horizontal global importance plot.
- `individual_explanations.csv`: source input value and SHAP contribution for each feature in each example, with probability/base-value and additivity diagnostics.
- `individual_explanation_example.png`: a SHAP waterfall plot for the first deterministic example.
- `explanation_metadata.json`: sample sizes, model/threshold/schema information, SHAP version, and additivity diagnostics.

## Reading the explanations

The global bar chart ranks features by the average absolute contribution over the selected development sample. Larger values mean that feature tends to move this fitted model's default probability farther from its background expectation. Mean signed SHAP values show the average direction across this sample, but can hide groups where effects go both ways.

For an individual example, a positive contribution moves the model output toward a higher default probability relative to the SHAP background expectation; a negative contribution moves it lower. The contributions add to the background value to reconstruct the model's positive-class probability (within the recorded numeric tolerance). They explain the model output, not why a borrower defaults.

## Limitations

SHAP is a feature-attribution method, not a causal method. Correlated inputs can share or shift attribution depending on the background and feature-perturbation assumptions. One-hot categorical contributions are summed across their encoded levels; this is useful for source-level reporting but does not make those features independent. The background is a small sample of this dataset's training partition, so explanations can change with a different sample or population. Results are exploratory and do not validate the model, establish fairness, or constitute an independent evaluation; current test metrics remain preliminary.
