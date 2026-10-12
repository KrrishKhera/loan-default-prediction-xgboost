"""CSV loading, schema validation, and reproducible dataset splitting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "loan_status"
NUMERIC_FEATURES = (
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
)
CATEGORICAL_FEATURES = (
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPECTED_COLUMNS = set(FEATURES) | {TARGET}


def validate_data(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate the dataset contract without changing or dropping any rows.

    Missing feature values are allowed because the preprocessing pipeline
    imputes them. Missing targets, unexpected columns, and invalid target
    labels are rejected.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("Dataset is empty")

    actual = set(frame.columns)
    missing_columns = sorted(EXPECTED_COLUMNS - actual)
    extra_columns = sorted(actual - EXPECTED_COLUMNS)
    if missing_columns or extra_columns:
        details = []
        if missing_columns:
            details.append(f"missing columns: {missing_columns}")
        if extra_columns:
            details.append(f"unexpected columns: {extra_columns}")
        raise ValueError("Invalid schema (" + "; ".join(details) + ")")

    non_numeric = [
        name for name in NUMERIC_FEATURES + (TARGET,)
        if not pd.api.types.is_numeric_dtype(frame[name])
    ]
    if non_numeric:
        raise TypeError(f"Expected numeric columns: {non_numeric}")

    missing_target = int(frame[TARGET].isna().sum())
    if missing_target:
        raise ValueError(f"Target {TARGET!r} contains {missing_target} missing values")
    labels = set(frame[TARGET].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Target {TARGET!r} must contain only 0 and 1; found {sorted(labels)}")

    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "target": TARGET,
        "target_counts": {int(k): int(v) for k, v in frame[TARGET].value_counts().sort_index().items()},
        "missing_values": {str(k): int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load a CSV and validate its schema and target without mutating it."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    frame = pd.read_csv(path)
    validate_data(frame)
    return frame


def _split_group_ids(
    group_ids: np.ndarray,
    group_labels: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split unique complete records, stratifying their associated target."""
    counts = pd.Series(group_labels).value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("At least two unique full records per target class are required to split")
    return train_test_split(
        group_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=group_labels,
    )


def split_data(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.20,
    validation_size: float = 0.20,
    random_state: int = 42,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Create deterministic train/validation/test partitions.

    Stratification is applied to unique predictor groups. Rows with identical
    predictors remain together, including conflicting-label cases, avoiding
    predictor duplication across partitions without dropping observations.
    Resulting row shares may differ slightly from requested proportions.
    """
    validate_data(frame)
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")

    # Group by predictors (not target), so identical borrower/loan inputs stay
    # together even if a future source revision contains conflicting labels.
    group_codes, _ = pd.factorize(
        pd.util.hash_pandas_object(frame.loc[:, list(FEATURES)], index=False)
    )
    group_labels_frame = pd.DataFrame({"group": group_codes, TARGET: frame[TARGET].to_numpy()})
    grouped = (
        group_labels_frame.groupby("group", sort=True)[TARGET]
        .agg(lambda values: int(values.mode().iloc[0]))
        .rename(TARGET)
        .reset_index()
    )
    group_ids = grouped["group"].to_numpy()
    group_labels = grouped[TARGET].to_numpy()

    trainval_groups, test_groups = _split_group_ids(
        group_ids, group_labels, test_size=test_size, random_state=random_state
    )
    labels_by_group = pd.Series(group_labels, index=group_ids)
    trainval_labels = labels_by_group.loc[trainval_groups].to_numpy()
    train_groups, validation_groups = _split_group_ids(
        trainval_groups,
        trainval_labels,
        test_size=validation_size / (1 - test_size),
        random_state=random_state,
    )

    row_groups = group_codes
    partition_masks = {
        "train": np.isin(row_groups, train_groups),
        "validation": np.isin(row_groups, validation_groups),
        "test": np.isin(row_groups, test_groups),
    }
    result: dict[str, pd.DataFrame | pd.Series] = {}
    for name, mask in partition_masks.items():
        part = frame.loc[mask]
        result[f"X_{name}"] = part.loc[:, list(FEATURES)].copy()
        result[f"y_{name}"] = part[TARGET].copy()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the source CSV")
    args = parser.parse_args()
    report = validate_data(load_data(args.csv))
    print(f"Validated {report['rows']} rows and {report['columns']} columns")
    print(f"Target counts: {report['target_counts']}")
    print(f"Missing values: {report['missing_values']}")
    print(f"Exact duplicate rows: {report['duplicate_rows']}")


if __name__ == "__main__":
    main()
