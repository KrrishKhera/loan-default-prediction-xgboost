"""Train, compare, select, and save loan-default classifiers."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from loan_default.data import TARGET, load_data, split_data
from loan_default.preprocessing import build_preprocessor


def build_model_grids(
    *, random_state: int, negative_to_positive_ratio: float
) -> dict[str, tuple[Any, list[dict[str, list[Any]]], str]]:
    """Return compact CV grids comparing unweighted and weighted options.

    Each search scores average precision (area under the precision-recall
    curve in its standard non-interpolated form), useful with an imbalanced
    binary target. All preprocessing remains inside each CV pipeline.
    """
    return {
        "logistic_regression": (
            LogisticRegression(solver="liblinear", max_iter=1000, random_state=random_state),
            [
                {"classifier__C": [0.1, 1.0], "classifier__class_weight": [None]},
                {"classifier__C": [0.1, 1.0], "classifier__class_weight": ["balanced"]},
            ],
            "None vs balanced class weights; C in {0.1, 1.0}",
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=160,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=1,
            ),
            [
                {
                    "classifier__max_depth": [None, 12],
                    "classifier__class_weight": [None],
                },
                {
                    "classifier__max_depth": [None, 12],
                    "classifier__class_weight": ["balanced_subsample"],
                },
            ],
            "Unweighted vs balanced_subsample; max_depth in {None, 12}",
        ),
        "xgboost": (
            XGBClassifier(
                n_estimators=220,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            ),
            [
                {
                    "classifier__max_depth": [3, 5],
                    "classifier__scale_pos_weight": [1.0],
                },
                {
                    "classifier__max_depth": [3, 5],
                    "classifier__scale_pos_weight": [negative_to_positive_ratio],
                },
            ],
            "scale_pos_weight=1 vs training-set negative/positive ratio; max_depth in {3, 5}",
        ),
    }


def _metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate discrimination and threshold metrics with TN/FP/FN/TP."""
    y_array = np.asarray(y_true, dtype=int)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_array, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_array, probabilities)),
        "pr_auc": float(average_precision_score(y_array, probabilities)),
        "precision": float(precision_score(y_array, predicted, zero_division=0)),
        "recall": float(recall_score(y_array, predicted, zero_division=0)),
        "f1": float(f1_score(y_array, predicted, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_f1_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> float:
    """Choose the validation threshold maximizing F1; ties favor precision."""
    y_array = np.asarray(y_true, dtype=int)
    precision, recall, thresholds = precision_recall_curve(y_array, probabilities)
    if thresholds.size == 0:
        raise ValueError("Cannot optimize a threshold without probability variation")
    denominators = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominators,
        out=np.zeros_like(denominators),
        where=denominators > 0,
    )
    best = np.flatnonzero(np.isclose(f1_values, f1_values.max()))
    # Larger threshold among equal-F1 choices predicts fewer positives and
    # therefore reduces false alarms (FP) at the cost of potentially more FN.
    return float(thresholds[best[-1]])


def train_and_evaluate(
    frame: pd.DataFrame,
    output_dir: str | Path,
    *,
    random_state: int = 42,
    cv_splits: int = 3,
    model_grids: dict[str, tuple[Any, list[dict[str, list[Any]]], str]] | None = None,
) -> dict[str, Any]:
    """Fit CV searches on train only, select using validation, test once.

    The selected fitted pipeline is still fit only on the training partition.
    Validation chooses the model and F1 threshold. The held-out test partition
    is only scored after those choices have been made.
    """
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    partitions = split_data(frame, random_state=random_state)
    X_train, y_train = partitions["X_train"], partitions["y_train"]
    X_val, y_val = partitions["X_validation"], partitions["y_validation"]
    X_test, y_test = partitions["X_test"], partitions["y_test"]
    assert isinstance(X_train, pd.DataFrame) and isinstance(y_train, pd.Series)
    assert isinstance(X_val, pd.DataFrame) and isinstance(y_val, pd.Series)
    assert isinstance(X_test, pd.DataFrame) and isinstance(y_test, pd.Series)

    train_negatives = int((y_train == 0).sum())
    train_positives = int((y_train == 1).sum())
    if train_positives == 0:
        raise ValueError("Training partition contains no positive target examples")
    class_ratio = train_negatives / train_positives
    grids = model_grids or build_model_grids(
        random_state=random_state,
        negative_to_positive_ratio=class_ratio,
    )
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    # Keep repeated predictor rows inside one CV fold as well as one outer
    # partition. This prevents exact duplicate records from validating copies
    # of themselves during hyperparameter search.
    train_cv_groups = pd.util.hash_pandas_object(X_train, index=False).to_numpy()

    searches: dict[str, GridSearchCV] = {}
    comparison_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    validation_probabilities: dict[str, np.ndarray] = {}

    for model_name, (estimator, param_grid, imbalance_note) in grids.items():
        pipeline = Pipeline(
            steps=[("preprocessor", build_preprocessor()), ("classifier", estimator)]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(X_train, y_train, groups=train_cv_groups)
        for params, mean_score, std_score, rank in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            search.cv_results_["std_test_score"],
            search.cv_results_["rank_test_score"],
        ):
            if model_name == "logistic_regression":
                strategy = "balanced class_weight" if params["classifier__class_weight"] else "unweighted"
            elif model_name == "random_forest":
                strategy = "balanced_subsample class_weight" if params["classifier__class_weight"] else "unweighted"
            else:
                ratio = float(params["classifier__scale_pos_weight"])
                strategy = "weighted scale_pos_weight" if ratio > 1 else "unweighted scale_pos_weight=1"
            candidate_rows.append(
                {
                    "model": model_name,
                    "imbalance_strategy": strategy,
                    "cv_mean_pr_auc": float(mean_score),
                    "cv_std_pr_auc": float(std_score),
                    "cv_rank_within_model": int(rank),
                    "params": json.dumps(params, sort_keys=True),
                }
            )
        probabilities = search.best_estimator_.predict_proba(X_val)[:, 1]
        validation_probabilities[model_name] = probabilities
        threshold = best_f1_threshold(y_val, probabilities)
        metrics_default = _metrics_at_threshold(y_val, probabilities, 0.5)
        metrics_tuned = _metrics_at_threshold(y_val, probabilities, threshold)
        comparison_rows.append(
            {
                "model": model_name,
                "cv_mean_pr_auc": float(search.best_score_),
                "cv_std_pr_auc": float(search.cv_results_["std_test_score"][search.best_index_]),
                "validation_roc_auc": metrics_default["roc_auc"],
                "validation_pr_auc": metrics_default["pr_auc"],
                "validation_precision_at_0_5": metrics_default["precision"],
                "validation_recall_at_0_5": metrics_default["recall"],
                "validation_f1_at_0_5": metrics_default["f1"],
                "validation_best_f1_threshold": threshold,
                "validation_precision_at_best_f1": metrics_tuned["precision"],
                "validation_recall_at_best_f1": metrics_tuned["recall"],
                "validation_best_f1": metrics_tuned["f1"],
                "validation_tn_at_best_f1": metrics_tuned["tn"],
                "validation_fp_at_best_f1": metrics_tuned["fp"],
                "validation_fn_at_best_f1": metrics_tuned["fn"],
                "validation_tp_at_best_f1": metrics_tuned["tp"],
                "imbalance_comparison": imbalance_note,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        searches[model_name] = search

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["validation_pr_auc", "cv_mean_pr_auc", "validation_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["model"])
    selected_search = searches[selected_name]
    selected_pipeline = selected_search.best_estimator_
    selected_probabilities = validation_probabilities[selected_name]
    selected_threshold = best_f1_threshold(y_val, selected_probabilities)

    threshold_rows: list[dict[str, Any]] = []
    threshold_candidates = sorted(set(np.round(np.arange(0.10, 0.951, 0.05), 2).tolist() + [0.5, selected_threshold]))
    for threshold in threshold_candidates:
        metrics = _metrics_at_threshold(y_val, selected_probabilities, threshold)
        threshold_rows.append(
            {
                "model": selected_name,
                "threshold": threshold,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "true_negatives": metrics["tn"],
                "false_positives_false_alarms": metrics["fp"],
                "false_negatives_missed_defaults": metrics["fn"],
                "true_positives": metrics["tp"],
            }
        )

    # Selection and threshold are now locked using CV-on-train and validation.
    # Test data is not used to alter any preceding decision.
    test_probabilities = selected_pipeline.predict_proba(X_test)[:, 1]
    test_at_selected_threshold = _metrics_at_threshold(y_test, test_probabilities, selected_threshold)
    test_at_default = _metrics_at_threshold(y_test, test_probabilities, 0.5)

    output_path = Path(output_dir)
    models_path = output_path / "models"
    reports_path = output_path / "reports"
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(reports_path / "model_comparison.csv", index=False)
    pd.DataFrame(candidate_rows).sort_values(
        ["model", "cv_rank_within_model", "cv_mean_pr_auc"],
        ascending=[True, True, False],
    ).to_csv(reports_path / "cv_candidate_comparison.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(reports_path / "threshold_analysis.csv", index=False)
    test_report = {
        "selected_model": selected_name,
        "selection_metric": "validation PR-AUC; hyperparameters tuned by training-only stratified CV PR-AUC",
        "decision_threshold": selected_threshold,
        "threshold_objective": "maximum validation F1; ties favor the higher threshold",
        "test_at_selected_threshold": test_at_selected_threshold,
        "test_at_default_threshold_0_5": test_at_default,
    }
    (reports_path / "test_metrics.json").write_text(json.dumps(test_report, indent=2) + "\n")

    artifact = {
        "pipeline": selected_pipeline,
        "decision_threshold": selected_threshold,
        "model_name": selected_name,
        "features": list(selected_pipeline.feature_names_in_),
        "target": TARGET,
        "random_state": random_state,
        "selection_metric": "validation PR-AUC",
        "threshold_objective": "maximum validation F1; ties favor the higher threshold",
        "best_params": selected_search.best_params_,
    }
    joblib.dump(artifact, models_path / "selected_model.joblib", compress=3)

    metadata = {
        "random_state": random_state,
        "cv_splits": cv_splits,
        "training_rows": int(len(y_train)),
        "validation_rows": int(len(y_val)),
        "test_rows": int(len(y_test)),
        "training_target_counts": {str(int(k)): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "validation_target_counts": {str(int(k)): int(v) for k, v in y_val.value_counts().sort_index().items()},
        "test_target_counts": {str(int(k)): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "best_params": selected_search.best_params_,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (reports_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "model_comparison": comparison,
        "validation_probabilities": validation_probabilities,
        "test_report": test_report,
        "output_dir": output_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/raw/credit_risk_dataset.csv")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-splits", type=int, default=3)
    args = parser.parse_args()

    result = train_and_evaluate(
        load_data(args.csv),
        args.output_dir,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
    )
    print("Validation model comparison:")
    print(result["model_comparison"].to_string(index=False))
    print(f"Selected model: {result['selected_model']}")
    print(f"Selected threshold: {result['selected_threshold']:.6f}")
    print("Final held-out test metrics:")
    print(json.dumps(result["test_report"], indent=2))


if __name__ == "__main__":
    main()
