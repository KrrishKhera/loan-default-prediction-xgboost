from __future__ import annotations

import numpy as np

from loan_default.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from loan_default.preprocessing import build_preprocessor
from test_data import sample_frame


def test_preprocessor_imputes_encodes_and_handles_unseen_categories():
    frame = sample_frame()
    frame.loc[0, "person_income"] = np.nan
    frame.loc[1, "loan_int_rate"] = np.nan
    frame.loc[2, "loan_intent"] = None
    transformer = build_preprocessor()
    transformed = transformer.fit_transform(frame.loc[:, list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)])

    assert transformed.shape[0] == len(frame)
    assert transformed.shape[1] > len(NUMERIC_FEATURES)
    assert np.isfinite(transformed.toarray() if hasattr(transformed, "toarray") else transformed).all()

    unseen = frame.iloc[[0]].loc[:, list(NUMERIC_FEATURES + CATEGORICAL_FEATURES)].copy()
    unseen.loc[:, "loan_intent"] = "UNSEEN_CATEGORY"
    result = transformer.transform(unseen)
    assert result.shape[0] == 1
