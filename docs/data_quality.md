# Data quality and feature policy

## Source and observed issues

The source is `data/raw/credit_risk_dataset.csv`; the raw file is kept unchanged and ignored by Git. The inspection found 32,581 rows and 12 columns, 165 exact duplicate rows, 895 missing `person_emp_length` values, and 3,116 missing `loan_int_rate` values. It also found five ages over 100 (maximum 144), two employment lengths greater than age (both 123 years), and nine zero `loan_percent_income` values despite positive amount and income. For 388 rows the recorded `loan_percent_income` differs from `loan_amnt / person_income` by more than 0.01; 68 differ by more than 0.05.

## Proposed treatment (no silent edits)

For the Stage 3 run, **no rows or feature values were removed, capped, flagged out of the model, or recomputed**. The existing pipeline imputes only values that are actually missing, with statistics learned inside each training fold. Therefore the five extreme age values, two employment-length-over-age records, nine zero income ratios, and ratio inconsistencies remain as recorded in all partitions. They warrant further source review before production use.

- **Exact duplicates:** retain every source row for now. The split helper groups rows by predictors, so duplicates cannot cross train/validation/test partitions even if a future source copy has inconsistent labels. Cross-validation also uses grouped folds. Before modeling, report their impact and decide whether the dataset documentation supports removing duplicate records. No rows are currently removed.
- **Extreme ages:** retain raw values and flag ages above 100 for review. If confirmed invalid, a documented cleaning step can replace them with missing values for pipeline imputation; do not cap or delete records without evidence.
- **Employment length:** median-impute genuinely missing values in the fitted preprocessing pipeline. Flag values greater than age (two cases) and the observed 123-year values. Confirm source meaning before treating these as invalid; a later cleaning policy may mark impossible values missing.
- **Income ratio:** retain the recorded feature for the initial baseline because it is available when terms are set. Preserve and report disagreements with amount divided by income; do not silently recompute or overwrite it. Investigate whether the source applies rounding or another convention before choosing a correction.

## Prediction point and leakage review

The stated use is **after loan terms are set and before repayment begins**. The initial baseline will use all 11 predictors other than `loan_status`, including:

- `loan_grade`: a pre-repayment underwriting term available at this prediction point; it is highly associated with outcome in this dataset and may encode prior underwriting judgment, but is not target leakage if timestamped before repayment.
- `loan_int_rate`: set as part of loan terms and available at this point. It may reflect underwriting decisions, which is appropriate for risk estimation after terms are set.
- `loan_percent_income`: available if income and amount have been recorded by this point. Its internal inconsistencies require investigation, but do not alone make it a post-outcome feature.

These decisions depend on confirming the source fields are recorded at origination and do not incorporate repayment-period updates. If the operational use changes to pre-underwriting application scoring, exclude `loan_grade` and `loan_int_rate`; compare their contribution in a sensitivity analysis. `loan_status` is the outcome and must never be an input.

## Metadata discrepancy

Kaggle metadata spells the credit-history field `cb_preson_cred_hist_length`; the actual CSV header is `cb_person_cred_hist_length`. The CSV spelling is used by the code.
