"""Preflight checks for a future independent final holdout.

This module validates schemas and duplicate-predictor separation only. It does
not load a model, generate predictions, or calculate evaluation metrics.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from loan_default.data import FEATURES, validate_data


def _predictor_group_hashes(frame: pd.DataFrame) -> pd.Series:
    """Return stable per-row hashes over predictors, excluding the target."""
    return pd.util.hash_pandas_object(frame.loc[:, list(FEATURES)], index=False)


def validate_independent_holdout(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
) -> dict[str, Any]:
    """Validate schema compatibility and reject predictor-group overlap.

    The check is label-independent: rows with the same predictor values are
    considered overlapping even when their target labels differ. Passing this
    check is a necessary integrity check, not proof that the holdout is new or
    was never inspected; source provenance must be established separately.
    """
    validate_data(development)
    validate_data(holdout)

    development_hashes = _predictor_group_hashes(development)
    holdout_hashes = _predictor_group_hashes(holdout)
    overlap = set(development_hashes) & set(holdout_hashes)
    if overlap:
        raise ValueError(
            "Holdout is not predictor-group independent: "
            f"{len(overlap)} predictor group(s) also occur in development data"
        )

    return {
        "development_rows": int(len(development)),
        "holdout_rows": int(len(holdout)),
        "development_predictor_groups": int(development_hashes.nunique()),
        "holdout_predictor_groups": int(holdout_hashes.nunique()),
        "overlapping_predictor_groups": 0,
        "feature_schema": list(FEATURES),
    }
