"""Exploratory SHAP explanations for the saved XGBoost model artifact.

This module never fits or writes a model. It explains a loaded fitted pipeline
using rows from the training partition only; it does not use the old test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loan_default.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    load_data,
    split_data,
)


def validate_artifact_schema(artifact: dict[str, Any]) -> None:
    """Reject an artifact whose documented features do not match this schema."""
    recorded = artifact.get("features")
    if recorded != list(FEATURES):
        raise ValueError(
            "Saved model feature schema does not match the project schema: "
            f"recorded={recorded!r}, expected={list(FEATURES)!r}"
        )
    if artifact.get("target") != TARGET or TARGET in recorded:
        raise ValueError("Artifact target metadata is invalid or target appears in features")
    pipeline = artifact.get("pipeline")
    if pipeline is None or not hasattr(pipeline, "named_steps"):
        raise ValueError("Artifact does not contain a fitted preprocessing/model pipeline")
    if "preprocessor" not in pipeline.named_steps or "classifier" not in pipeline.named_steps:
        raise ValueError("Artifact pipeline must contain preprocessor and classifier steps")


def encoded_source_features(preprocessor: Any) -> list[str]:
    """Map each transformed numeric/one-hot column back to its raw feature."""
    if not hasattr(preprocessor, "named_transformers_"):
        raise ValueError("Preprocessor is not fitted")
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["onehot"]
    mapped = list(NUMERIC_FEATURES)
    for feature, categories in zip(CATEGORICAL_FEATURES, encoder.categories_):
        mapped.extend([feature] * len(categories))
    expected = len(preprocessor.get_feature_names_out())
    if len(mapped) != expected:
        raise ValueError(f"Encoded feature mapping has {len(mapped)} entries; expected {expected}")
    return mapped


def aggregate_source_feature_values(
    encoded_values: np.ndarray,
    encoded_to_source: Sequence[str],
    *,
    source_features: Sequence[str] = FEATURES,
) -> np.ndarray:
    """Sum encoded-column SHAP contributions back to source feature values."""
    values = np.asarray(encoded_values)
    if values.ndim != 2 or values.shape[1] != len(encoded_to_source):
        raise ValueError("encoded_values must have shape (rows, encoded features)")
    positions = {name: index for index, name in enumerate(source_features)}
    if any(name not in positions for name in encoded_to_source):
        raise ValueError("Encoded mapping contains a feature absent from source_features")
    grouped = np.zeros((values.shape[0], len(source_features)), dtype=float)
    for encoded_index, source_name in enumerate(encoded_to_source):
        grouped[:, positions[source_name]] += values[:, encoded_index]
    return grouped


def _dense(matrix: Any) -> np.ndarray:
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def validate_raw_features(rows: pd.DataFrame) -> pd.DataFrame:
    """Validate and order raw borrower inputs for the saved model pipeline."""
    if not isinstance(rows, pd.DataFrame):
        raise TypeError("Prediction inputs must be a pandas DataFrame")
    missing = sorted(set(FEATURES) - set(rows.columns))
    extra = sorted(set(rows.columns) - set(FEATURES))
    if missing or extra:
        raise ValueError(f"Invalid prediction schema; missing={missing}, unexpected={extra}")
    return rows.loc[:, list(FEATURES)].copy()


def build_shap_explainer(artifact: dict[str, Any], background_rows: pd.DataFrame) -> Any:
    """Create a probability-scale TreeExplainer from raw background rows."""
    validate_artifact_schema(artifact)
    background = validate_raw_features(background_rows)
    preprocessor = artifact["pipeline"].named_steps["preprocessor"]
    classifier = artifact["pipeline"].named_steps["classifier"]
    background_encoded = _dense(preprocessor.transform(background))
    return shap.TreeExplainer(
        classifier,
        data=background_encoded,
        feature_perturbation="interventional",
        model_output="probability",
    )


def explain_instances(
    artifact: dict[str, Any],
    rows: pd.DataFrame,
    explainer: Any,
) -> dict[str, Any]:
    """Explain raw borrower rows and return probabilities plus source attributions.

    The saved fitted preprocessor and classifier are used end to end. Callers
    may reuse one explainer/background for several prediction rows.
    """
    validate_artifact_schema(artifact)
    raw_rows = validate_raw_features(rows)
    if raw_rows.empty:
        raise ValueError("At least one prediction row is required")
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    encoded_to_source = encoded_source_features(preprocessor)
    encoded_rows = _dense(preprocessor.transform(raw_rows))
    explanation = explainer(encoded_rows, check_additivity=True)
    encoded_values = _positive_class_values(explanation.values)
    source_values = aggregate_source_feature_values(encoded_values, encoded_to_source)
    base_value = _positive_base_value(explanation.base_values)
    probabilities = pipeline.predict_proba(raw_rows)[:, 1]
    reconstructed = base_value + source_values.sum(axis=1)
    additivity_error = np.abs(reconstructed - probabilities)
    if not np.allclose(reconstructed, probabilities, atol=2e-4, rtol=2e-4):
        raise RuntimeError(
            "Grouped SHAP values do not reconstruct positive-class probabilities; "
            f"maximum absolute difference={additivity_error.max():.6g}"
        )
    return {
        "contributions": pd.DataFrame(source_values, index=raw_rows.index, columns=list(FEATURES)),
        "probabilities": probabilities,
        "base_value": base_value,
        "additivity_errors": additivity_error,
    }


def _positive_class_values(values: Any) -> np.ndarray:
    """Normalize SHAP's common binary classifier output shapes to (rows, cols)."""
    if isinstance(values, list):
        if len(values) == 2:
            values = values[1]
        elif len(values) == 1:
            values = values[0]
        else:
            raise ValueError(f"Expected binary SHAP output, got {len(values)} class arrays")
    result = np.asarray(values)
    if result.ndim == 3:
        if result.shape[-1] == 2:
            result = result[:, :, 1]
        elif result.shape[0] == 2:
            result = result[1]
    if result.ndim != 2:
        raise ValueError(f"Expected 2D positive-class SHAP values; got shape {result.shape}")
    return result


def _positive_base_value(value: Any) -> float:
    base = np.asarray(value)
    if base.ndim == 0:
        return float(base)
    flat = base.ravel()
    return float(flat[1] if flat.size == 2 else flat[0])


def explain_saved_model(
    *,
    csv_path: str | Path,
    artifact_path: str | Path,
    output_dir: str | Path,
    random_state: int = 42,
    background_size: int = 80,
    explanation_size: int = 300,
    individual_count: int = 3,
) -> dict[str, Any]:
    """Generate exploratory global and local explanations from training rows.

    The old test partition is not scored or passed to SHAP. The model artifact
    is loaded read-only; no estimator or preprocessing step is refit.
    """
    if background_size < 1 or explanation_size < 1 or individual_count < 1:
        raise ValueError("Sample sizes must be positive")

    artifact = joblib.load(artifact_path)
    validate_artifact_schema(artifact)
    frame = load_data(csv_path)
    partitions = split_data(frame, random_state=random_state)
    X_train = partitions["X_train"]
    assert isinstance(X_train, pd.DataFrame)
    # All explanation rows and the SHAP background come from development
    # training data, never from validation or the previously evaluated test.
    n_background = min(background_size, len(X_train))
    n_explain = min(explanation_size, len(X_train))
    background = X_train.sample(n=n_background, random_state=random_state)
    explanation_rows = X_train.sample(n=n_explain, random_state=random_state + 1)
    n_individual = min(individual_count, len(explanation_rows))

    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    encoded_to_source = encoded_source_features(preprocessor)
    explainer = build_shap_explainer(artifact, background)
    individual_explanations = explain_instances(artifact, explanation_rows, explainer)
    source_values = individual_explanations["contributions"].to_numpy()
    base_value = individual_explanations["base_value"]
    probabilities = individual_explanations["probabilities"]
    additivity_error = individual_explanations["additivity_errors"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    importance = pd.DataFrame(
        {
            "feature": list(FEATURES),
            "mean_abs_shap_probability": np.mean(np.abs(source_values), axis=0),
            "mean_shap_probability": np.mean(source_values, axis=0),
            "sample_rows": n_explain,
        }
    ).sort_values("mean_abs_shap_probability", ascending=False)
    importance.to_csv(output_path / "global_feature_importance.csv", index=False)

    plot_data = importance.sort_values("mean_abs_shap_probability")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(plot_data["feature"], plot_data["mean_abs_shap_probability"], color="#2878B5")
    ax.set_xlabel("Mean absolute SHAP value (default-probability contribution)")
    ax.set_title("Exploratory global SHAP importance — existing XGBoost model")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path / "global_feature_importance.png", dpi=160)
    plt.close(fig)

    local_rows: list[dict[str, Any]] = []
    for row_index in range(n_individual):
        source_contributions = source_values[row_index]
        row_probability = float(probabilities[row_index])
        row_error = float(additivity_error[row_index])
        for feature_index, feature in enumerate(FEATURES):
            contribution = float(source_contributions[feature_index])
            local_rows.append(
                {
                    "example_id": row_index + 1,
                    "feature": feature,
                    "input_value": str(explanation_rows.iloc[row_index][feature]),
                    "shap_value_probability": contribution,
                    "direction": "raises default probability" if contribution > 0 else "lowers default probability",
                    "base_probability": base_value,
                    "predicted_default_probability": row_probability,
                    "additivity_error": row_error,
                }
            )
    pd.DataFrame(local_rows).to_csv(output_path / "individual_explanations.csv", index=False)

    # Waterfall uses aggregated original features so categories are explained
    # as one borrower feature rather than a list of one-hot columns.
    local_explanation = shap.Explanation(
        values=source_values[0],
        base_values=base_value,
        data=explanation_rows.iloc[0].to_numpy(dtype=object),
        feature_names=list(FEATURES),
    )
    shap.plots.waterfall(local_explanation, max_display=len(FEATURES), show=False)
    plt.title("Exploratory individual SHAP explanation — existing XGBoost model")
    plt.tight_layout()
    plt.savefig(output_path / "individual_explanation_example.png", dpi=160, bbox_inches="tight")
    plt.close()

    metadata = {
        "status": "exploratory explanation of the existing saved model",
        "model_name": artifact["model_name"],
        "decision_threshold": artifact["decision_threshold"],
        "feature_schema": artifact["features"],
        "target_excluded_from_features": artifact["target"] not in artifact["features"],
        "data_partition": "training rows from deterministic development split only",
        "random_state": random_state,
        "background_rows": n_background,
        "global_explanation_rows": n_explain,
        "individual_examples": n_individual,
        "shap_version": shap.__version__,
        "explainer": "TreeExplainer, interventional, model_output=probability",
        "mean_absolute_additivity_error": float(additivity_error.mean()),
        "maximum_absolute_additivity_error": float(additivity_error.max()),
        "interpretation": "Attributions describe this fitted model's output relative to its background sample; they are not causal effects.",
    }
    (output_path / "explanation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"importance": importance, "metadata": metadata, "output_dir": output_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/raw/credit_risk_dataset.csv")
    parser.add_argument("--artifact", default="models/selected_model.joblib")
    parser.add_argument("--output-dir", default="reports/exploratory_shap")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--background-size", type=int, default=80)
    parser.add_argument("--explanation-size", type=int, default=300)
    parser.add_argument("--individual-count", type=int, default=3)
    args = parser.parse_args()
    result = explain_saved_model(
        csv_path=args.csv,
        artifact_path=args.artifact,
        output_dir=args.output_dir,
        random_state=args.random_state,
        background_size=args.background_size,
        explanation_size=args.explanation_size,
        individual_count=args.individual_count,
    )
    print(f"Wrote exploratory explanations to {result['output_dir']}")
    print(result["importance"].to_string(index=False))


if __name__ == "__main__":
    main()
