from __future__ import annotations

import pandas as pd
import pytest

from loan_default.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_data,
    split_data,
    validate_data,
)


def sample_frame() -> pd.DataFrame:
    rows = []
    for i in range(40):
        row = {column: float(i + 1) for column in NUMERIC_FEATURES}
        row.update({column: f"category_{i % 3}" for column in CATEGORICAL_FEATURES})
        row[TARGET] = i % 2
        rows.append(row)
    return pd.DataFrame(rows)


def test_load_data_and_validate_schema(tmp_path):
    frame = sample_frame()
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)

    loaded = load_data(path)

    assert loaded.shape == frame.shape
    assert validate_data(loaded)["target_counts"] == {0: 20, 1: 20}


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "missing.csv")


def test_schema_and_target_errors_are_reported():
    frame = sample_frame()
    with pytest.raises(ValueError, match="missing columns"):
        validate_data(frame.drop(columns=["loan_grade"]))

    frame.loc[0, TARGET] = 2
    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_data(frame)


def test_missing_features_are_allowed_but_missing_target_is_not():
    frame = sample_frame()
    frame.loc[0, "person_income"] = float("nan")
    validate_data(frame)
    frame.loc[1, TARGET] = float("nan")
    with pytest.raises(ValueError, match="missing values"):
        validate_data(frame)


def test_split_is_reproducible_stratified_and_keeps_duplicates_together():
    frame = sample_frame()
    # Add one exact duplicate; duplicate rows are retained but grouped together.
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    first = split_data(frame, random_state=17)
    second = split_data(frame, random_state=17)

    for key in first:
        pd.testing.assert_frame_equal(first[key], second[key]) if isinstance(first[key], pd.DataFrame) else pd.testing.assert_series_equal(first[key], second[key])

    total = sum(len(first[f"y_{part}"]) for part in ("train", "validation", "test"))
    assert total == len(frame)
    for part in ("train", "validation", "test"):
        assert set(first[f"y_{part}"].unique()) == {0, 1}
    duplicate = frame.iloc[0]
    placements = []
    for part in ("train", "validation", "test"):
        x = first[f"X_{part}"]
        y = first[f"y_{part}"]
        placements.extend(
            [part] * int(((x == duplicate[list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)]).all(axis=1) & (y == duplicate[TARGET])).sum())
        )
    assert len(placements) == 2
    assert len(set(placements)) == 1


def test_split_keeps_identical_predictors_together_if_labels_conflict():
    frame = sample_frame()
    duplicate = frame.iloc[[0]].copy()
    duplicate.loc[:, TARGET] = 1 - int(duplicate[TARGET].iloc[0])
    frame = pd.concat([frame, duplicate], ignore_index=True)
    partitions = split_data(frame, random_state=23)

    target_features = frame.iloc[0][list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)]
    placements = []
    for part in ("train", "validation", "test"):
        matches = partitions[f"X_{part}"].eq(target_features).all(axis=1)
        placements.extend([part] * int(matches.sum()))
    assert len(placements) == 2
    assert len(set(placements)) == 1
