# Training and evaluation protocol

## Data partitions and leakage controls

`loan_default.data.split_data` creates deterministic 60/20/20 development partitions (row proportions are approximate because repeated predictor groups remain together). Stratification is performed over unique predictor groups; rows with identical predictors stay together even if labels conflict. All records are retained. The Stage 3 test partition was evaluated multiple times and is now considered preliminary development diagnostic data, not a pristine final holdout.

Each candidate is a scikit-learn `Pipeline` containing the existing `build_preprocessor()` and a classifier. `GridSearchCV` fits imputers, scaling, and one-hot categories within each training fold. CV uses shuffled stratified group folds with a fixed seed; repeated predictor groups cannot straddle a fold boundary. Search scoring is average precision, commonly called PR-AUC; this emphasizes retrieval quality for the minority default class.

## Candidate models and imbalance comparison

- **Logistic Regression:** compares unweighted and `class_weight="balanced"`, with `C` in `{0.1, 1.0}`.
- **Random Forest:** compares unweighted and `class_weight="balanced_subsample"`, with maximum depth in `{None, 12}`.
- **XGBoost:** compares `scale_pos_weight=1` with the negative/positive ratio calculated from training rows only; maximum depth in `{3, 5}`.

These are compact, controlled comparisons rather than a broad expensive search. No weighting or resampling is applied automatically. Model family and hyperparameters are chosen using training-only CV followed by validation PR-AUC. The final threshold is selected on validation by maximum F1; ties favor the higher threshold.

## Threshold interpretation

At a given threshold, a default borrower predicted as non-default is a false negative (FN, a missed default). A non-default borrower predicted as default is a false positive (FP, a false alarm). Lowering the threshold generally catches more defaults (higher recall, fewer FN) while flagging more non-default borrowers (lower precision, more FP). The project has no agreed business costs for these errors, so maximum F1 is used as a transparent initial balance, not as a universal lending policy. `reports/threshold_analysis.csv` shows the selected model's validation trade-off over a range of thresholds.

## Artifacts and outputs

The historical Stage 3 command `PYTHONPATH=src python -m loan_default.training` writes (and can overwrite) the following. Do not rerun it as a final evaluation; it uses the already-exposed partition and rewrites model/report outputs:

- `reports/model_comparison.csv`: CV and validation metrics, default-threshold results, best validation F1 threshold and parameters for all model families.
- `reports/cv_candidate_comparison.csv`: training-only CV scores for every weighted/unweighted candidate and hyperparameter setting.
- `reports/threshold_analysis.csv`: validation precision, recall, F1, and confusion counts across thresholds for the selected family.
- `reports/test_metrics.json`: historical preliminary test-partition metrics; preserve as-is unless a separately approved documentation change is requested.
- `reports/run_metadata.json`: seed, split sizes, class counts, selected parameters, and library versions.
- `models/selected_model.joblib`: selected fitted preprocessing+classifier pipeline and threshold/metadata. Model artifacts are excluded from Git.

The selected estimator remains fit on the training partition only. Validation chose the family and threshold. The old test metrics were not used to tune the reported candidate, but repeated evaluation means they are not an independent final estimate. A future untouched holdout requires genuinely new compatible data; repartitioning this same fully used CSV cannot restore test independence.

See [independent_holdout.md](independent_holdout.md) for the future holdout protocol. The preflight helper checks schema and predictor-group disjointness only; it does not establish data provenance or calculate model metrics.

## Feature timing and data-quality decisions

The prediction point is after terms are set and before repayment begins. `loan_grade`, `loan_int_rate`, and `loan_percent_income` are included on the assumption that their values are available at origination and do not contain repayment-period information. Confirm timestamps and provenance before treating this assumption as operational fact. See [data_quality.md](data_quality.md) for the duplicate, extreme-value, ratio-inconsistency, and metadata-typo treatment; raw rows are not silently modified or dropped.
