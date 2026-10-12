from __future__ import annotations

import pytest

from loan_default.data import TARGET
from loan_default.holdout import validate_independent_holdout
from test_data import sample_frame


def test_preflight_accepts_synthetic_disjoint_predictor_groups():
    synthetic = sample_frame()
    development = synthetic.iloc[:30].copy()
    holdout = synthetic.iloc[30:].copy()

    result = validate_independent_holdout(development, holdout)

    assert result["development_rows"] == 30
    assert result["holdout_rows"] == 10
    assert result["overlapping_predictor_groups"] == 0
    assert TARGET not in result["feature_schema"]


def test_preflight_rejects_duplicate_predictors_even_when_labels_conflict():
    development = sample_frame().iloc[:30].copy()
    holdout = development.iloc[[0]].copy()
    holdout.loc[:, TARGET] = 1 - int(holdout[TARGET].iloc[0])

    with pytest.raises(ValueError, match="predictor-group independent"):
        validate_independent_holdout(development, holdout)


def test_preflight_rejects_incompatible_schema():
    synthetic = sample_frame()
    development = synthetic.iloc[:30].copy()
    holdout = synthetic.iloc[30:].drop(columns=["loan_grade"])

    with pytest.raises(ValueError, match="missing columns"):
        validate_independent_holdout(development, holdout)
