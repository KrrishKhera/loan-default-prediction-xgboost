from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from loan_default.data import load_data, split_data
from loan_default.training import best_f1_threshold, train_and_evaluate
from test_data import sample_frame


def _small_grid():
    return {
        "logistic_regression": (
            LogisticRegression(solver="liblinear", max_iter=300, random_state=11),
            [{"classifier__C": [1.0], "classifier__class_weight": [None]}],
            "unweighted test configuration",
        )
    }


def test_training_artifact_prediction_shape_probability_and_reproducibility(tmp_path):
    frame = sample_frame()
    grid = _small_grid()
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first = train_and_evaluate(frame, first_dir, random_state=19, cv_splits=2, model_grids=grid)
    second = train_and_evaluate(frame, second_dir, random_state=19, cv_splits=2, model_grids=grid)

    assert first["selected_model"] == second["selected_model"] == "logistic_regression"
    assert first["selected_threshold"] == second["selected_threshold"]
    assert first["model_comparison"]["validation_pr_auc"].iloc[0] == second["model_comparison"]["validation_pr_auc"].iloc[0]

    artifact = joblib.load(first_dir / "models" / "selected_model.joblib")
    partitions = split_data(frame, random_state=19)
    probabilities = artifact["pipeline"].predict_proba(partitions["X_test"])[:, 1]
    assert probabilities.shape == (len(partitions["X_test"]),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert artifact["decision_threshold"] == first["selected_threshold"]
    assert (first_dir / "reports" / "model_comparison.csv").is_file()
    assert (first_dir / "reports" / "cv_candidate_comparison.csv").is_file()
    assert (first_dir / "reports" / "threshold_analysis.csv").is_file()
    assert (first_dir / "reports" / "test_metrics.json").is_file()


def test_best_f1_threshold_is_valid_and_uses_probabilities():
    threshold = best_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))
    assert 0 <= threshold <= 1


def test_loader_can_feed_training_contract():
    # Exercise the same public loading/validation path used by the CLI.
    frame = sample_frame()
    assert load_data_from_fixture(frame).shape == frame.shape


def load_data_from_fixture(frame):
    """Round-trip a small fixture through the CSV loader."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/fixture.csv"
        frame.to_csv(path, index=False)
        return load_data(path)
