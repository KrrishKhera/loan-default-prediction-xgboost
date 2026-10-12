# Future independent holdout protocol

## Current status

No final holdout exists in this project. The only available CSV, `data/raw/credit_risk_dataset.csv`, has already been used across model development and repeated test-partition evaluations. Re-splitting it cannot recover an untouched holdout. Existing test metrics remain preliminary development diagnostics.

Use genuinely new, compatible loan records with the same `loan_status` outcome definition and the same feature schema. Prefer a later origination cohort whose outcomes have matured, or another independently collected dataset. Record provenance and collection/outcome timing before accepting it. The workspace location `data/final_holdout/` is Git-ignored for the future file; it has not been created or populated by this change.

## Reserve before development decisions

1. Place the new source file in `data/final_holdout/` without opening it for EDA or model analysis. Keep the original development dataset as development data.
2. Validate the exact required columns and target with `validate_data`. Reject missing/extra columns, invalid target labels, or incompatible target meaning.
3. Check predictor-group hashes against all development data using `validate_independent_holdout`. The check hashes only the 11 approved predictors, not `loan_status`, so identical predictors are detected even if labels conflict. Any overlap blocks the holdout; resolve by obtaining more new data or moving entire predictor groups to development before freezing.
4. Passing the hash check is necessary but not sufficient: confirm externally that the records are genuinely new and that their labels were not previously inspected. Feature hashes cannot detect the same borrower represented by different values or undisclosed prior access.
5. Before the final evaluation, freeze and record the ordered feature schema, target definition, selected model artifact SHA-256, decision threshold, code revision, dependency versions, and evaluation metric definitions. The current feature schema is the existing 11-column `FEATURES` allowlist; `loan_status` is target-only. The current candidate threshold is 0.4504756033420563 unless a later development process deliberately selects a different candidate and freezes it before holdout access.
6. Run one final evaluation at the frozen threshold. Do not tune, calculate alternate-threshold test metrics, or change the model in response to holdout results. Write results to a new `reports/final_holdout_metrics.json`; do not overwrite `reports/test_metrics.json` or current artifacts. Any subsequent model changes require another independent holdout for a new final claim.

The current `validate_independent_holdout` helper implements only steps 2–3. It performs no model loading, prediction, metric calculation, reservation, or provenance verification. Those later actions require a separately reviewed one-shot evaluator after genuinely new data is supplied.

## Future commands

After new data has been supplied and the protocol implementation approved, the planned schema/overlap preflight is:

```bash
PYTHONPATH=src python -m loan_default.holdout_preflight \
  --development data/raw/credit_risk_dataset.csv \
  --holdout data/final_holdout/credit_risk_final_holdout.csv
```

This is a proposed future CLI; only the Python preflight function and synthetic tests exist now. It must not produce predictions or metrics. A separate final-evaluation command will be specified and reviewed after the holdout is supplied and the candidate is frozen. Do not run the historical `loan_default.training` command for this purpose.
