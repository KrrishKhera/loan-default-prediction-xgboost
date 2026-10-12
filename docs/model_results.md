# Stage 3 model results (preliminary diagnostics)

> **Evaluation status:** The reported test-partition metrics are preliminary development diagnostics, not an untouched final holdout result. The partition was evaluated multiple times during implementation, including repeated full training/evaluation runs and an artifact prediction smoke check. Preserve these values as historical results, but do not use them as an independent estimate of generalization or as a basis for further tuning.

Run on the unchanged 32,581-row source CSV with random seed 42 and three stratified group CV folds. Split sizes were 19,544 train, 6,514 validation, and 6,523 test rows. All imputers, scaling, encoding, and model fits were learned on training folds / the training partition only. Predictor duplicates stayed within a single outer partition and CV fold. No anomalous source values were changed.

The three model families were tuned by training-only CV average precision. The family was then selected by validation PR-AUC. Thresholds below were found on validation by maximum F1.

| Model (best CV candidate) | CV PR-AUC mean ± SD | Validation ROC-AUC | Validation PR-AUC | Precision / recall / F1 at 0.5 | Best F1 threshold | Precision / recall / F1 at tuned threshold |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (`max_depth=5`, `scale_pos_weight=1`) | 0.8920 ± 0.0050 | 0.9444 | 0.8978 | 0.9737 / 0.7303 / 0.8346 | 0.4505 | 0.9615 / 0.7394 / 0.8360 |
| Random Forest (unweighted, unlimited depth) | 0.8772 ± 0.0049 | 0.9287 | 0.8842 | 0.9716 / 0.7218 / 0.8283 | 0.4305 | 0.9545 / 0.7387 / 0.8329 |
| Logistic Regression (`C=1`, unweighted) | 0.7190 ± 0.0094 | 0.8778 | 0.7343 | 0.7668 / 0.5789 / 0.6597 | 0.3544 | 0.6761 / 0.7056 / 0.6906 |

## Selection and class imbalance

XGBoost was selected because it had the highest validation PR-AUC (0.8978 versus 0.8842 for Random Forest and 0.7343 for Logistic Regression). It also had the highest training CV PR-AUC among the best candidates. Its selected CV configuration was unweighted (`scale_pos_weight=1`). The best CV candidate in each family was unweighted; the weighted alternatives did not improve CV average precision in this run. Full candidate scores are in [`../reports/cv_candidate_comparison.csv`](../reports/cv_candidate_comparison.csv).

At the validation cutoff 0.5, the selected XGBoost model produced 28 false alarms and missed 383 defaults. At the validation F1 threshold 0.4505, it produced 42 false alarms and missed 370 defaults. The lower threshold therefore caught 13 additional defaults at the cost of 14 additional false alarms; validation F1 moved from 0.8346 to 0.8360. This is a small gain, and the operational cost of the two error types has not been specified, so 0.4505 is a transparent starting threshold rather than a business-approved policy.

## Preliminary test-partition diagnostics

The historical test-partition values at threshold 0.4505 were ROC-AUC **0.9405**, PR-AUC **0.8954**, precision **0.9613**, recall **0.7321**, and F1 **0.8312**. The recorded confusion counts were TN=5,055, FP=42, FN=382, TP=1,044. At threshold 0.5, recorded F1 was 0.8294 (29 false alarms and 395 missed defaults). These values are retained for traceability but are preliminary because the test partition was evaluated multiple times. They must not be described as a one-time final evaluation.

## Artifacts and limitations

- Detailed family results: [`../reports/model_comparison.csv`](../reports/model_comparison.csv)
- Imbalance / parameter candidate CV results: [`../reports/cv_candidate_comparison.csv`](../reports/cv_candidate_comparison.csv)
- Validation threshold sweep: [`../reports/threshold_analysis.csv`](../reports/threshold_analysis.csv)
- Test metrics: [`../reports/test_metrics.json`](../reports/test_metrics.json)
- Reproducibility metadata: [`../reports/run_metadata.json`](../reports/run_metadata.json)
- Selected preprocessing+XGBoost pipeline and threshold: `models/selected_model.joblib` (ignored by Git)

The dataset is a static sample; real-world calibration, temporal validation, subgroup performance, and the cost of false negatives versus false alarms have not been assessed. Feature timestamps and provenance for `loan_grade`, `loan_int_rate`, and `loan_percent_income` still need confirmation. Ages above 100, employment lengths above age, zero income ratios, and inconsistent income ratios were retained unchanged, and may affect results. No claim about production lending suitability is implied.
