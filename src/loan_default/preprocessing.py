"""Reusable scikit-learn preprocessing for numeric and categorical fields."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_default.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build missing-value imputation, scaling, and categorical encoding.

    Numeric missing values use the training-fold median; categorical missing
    values use the training-fold most frequent category. One-hot encoding
    ignores categories not seen during fit. Scaling supports linear baselines;
    tree models can use this same interface or a later tree-specific variant.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )
