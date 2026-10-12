"""Read-only loaders and validated summaries for existing project artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import streamlit as st
import pandas as pd
from loan_default.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

def read_csv_report(path: str | Path) -> pd.DataFrame | None:
    """Load a report table if it exists and is readable."""
    report_path = Path(path)
    if not report_path.is_file():
        return None
    try:
        frame = pd.read_csv(report_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None
    return frame if not frame.empty else None


def read_json_report(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON report if it exists and has an object at its root."""
    report_path = Path(path)
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def validate_borrower_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Check exact field presence and reject target/unknown keys before HTTP."""
    expected = set(FEATURES)
    actual = set(payload)
    missing, unexpected = expected - actual, actual - expected
    if missing or unexpected:
        raise ValueError(
            f"Invalid borrower payload; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return {feature: payload[feature] for feature in FEATURES}


def summarize_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize a compatible dataset without fitting or evaluating a model."""
    required = set(FEATURES) | {TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty")

    target_counts = frame[TARGET].value_counts(dropna=False).sort_index()

    return {
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "features": len(FEATURES),
        "missing_cells": int(frame.loc[:, list(FEATURES)].isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "target_counts": {
            int(key): int(value) for key, value in target_counts.items()
        },
        "target_percentages": {
            int(key): float(value / len(frame) * 100)
            for key, value in target_counts.items()
        },
        "numeric_summary": frame.loc[:, list(NUMERIC_FEATURES)].describe().to_dict(),
        "categorical_values": {
            feature: sorted(frame[feature].dropna().astype(str).unique().tolist())
            for feature in CATEGORICAL_FEATURES
        },
    }

