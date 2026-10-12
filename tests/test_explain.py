from __future__ import annotations

import numpy as np
import pytest

from loan_default.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from loan_default.explain import aggregate_source_feature_values, validate_artifact_schema


def test_aggregate_source_feature_values_sums_one_hot_contributions():
    mapping = ["person_age", "loan_grade", "loan_grade", "loan_intent"]
    values = np.array([[0.1, 0.02, -0.03, 0.05], [-0.2, 0.04, 0.01, -0.01]])
    grouped = aggregate_source_feature_values(values, mapping, source_features=["person_age", "loan_grade", "loan_intent"])

    np.testing.assert_allclose(grouped, [[0.1, -0.01, 0.05], [-0.2, 0.05, -0.01]])


def test_validate_artifact_schema_accepts_saved_pipeline_contract():
    class PipelineStub:
        named_steps = {"preprocessor": object(), "classifier": object()}

    validate_artifact_schema(
        {"features": list(FEATURES), "target": TARGET, "pipeline": PipelineStub()}
    )


def test_validate_artifact_schema_rejects_target_or_schema_mismatch():
    class PipelineStub:
        named_steps = {"preprocessor": object(), "classifier": object()}

    bad_features = list(FEATURES[:-1]) + [TARGET]
    with pytest.raises(ValueError, match="feature schema"):
        validate_artifact_schema({"features": bad_features, "target": TARGET, "pipeline": PipelineStub()})


def test_preprocessor_encoded_feature_mapping_uses_fitted_synthetic_pipeline():
    from loan_default.explain import encoded_source_features
    from loan_default.preprocessing import build_preprocessor
    from test_data import sample_frame

    synthetic = sample_frame().loc[:, list(FEATURES)]
    transformer = build_preprocessor()
    transformer.fit(synthetic)
    mapping = encoded_source_features(transformer)

    assert mapping[: len(NUMERIC_FEATURES)] == list(NUMERIC_FEATURES)
    assert set(mapping[len(NUMERIC_FEATURES):]).issubset(set(CATEGORICAL_FEATURES))
    assert len(mapping) == len(transformer.get_feature_names_out())


def test_raw_prediction_schema_rejects_target_and_orders_features():
    import pandas as pd

    from loan_default.explain import validate_raw_features

    row = {feature: [1] for feature in FEATURES}
    unordered = pd.DataFrame(dict(reversed(list(row.items()))))
    ordered = validate_raw_features(unordered)
    assert list(ordered.columns) == list(FEATURES)

    with_target = unordered.assign(**{TARGET: [0]})
    with pytest.raises(ValueError, match="unexpected"):
        validate_raw_features(with_target)
