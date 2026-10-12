"""Saved model comparison and evaluation report section."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.charts import (
    confusion_chart,
    cv_chart,
    threshold_tradeoff_chart,
    validation_ranking_chart,
    validation_threshold_chart,
)
from app.theme import page_heading


def render(reports: dict[str, Any], health: dict[str, Any] | None) -> None:
    dark = st.session_state.get("appearance") == "Dark"
    page_heading(
        "Model analytics",
        "Compare saved cross-validation and validation results, with historical test diagnostics clearly separated.",
        eyebrow="MODEL PERFORMANCE",
        service_ready=bool(health and health.get("model_loaded")),
    )
    comparison = reports.get("comparison")
    if not isinstance(comparison, pd.DataFrame) or comparison.empty:
        st.warning("The model comparison report is missing or unreadable. No performance metrics are being inferred.")
        return

    st.markdown("#### Training-only grouped cross-validation")
    st.caption("Mean PR-AUC ± standard deviation across the saved grouped folds. This is separate from validation and test evaluation.")
    st.plotly_chart(cv_chart(comparison, dark=dark), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Validation results")
    st.caption("Validation selected the model family and decision threshold. These are development results, not a fresh independent holdout.")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.plotly_chart(validation_ranking_chart(comparison, dark=dark), width="stretch", config={"displayModeBar": False})
    with col2:
        st.plotly_chart(validation_threshold_chart(comparison, dark=dark), width="stretch", config={"displayModeBar": False})

    selected = "xgboost"
    rowset = comparison.loc[comparison["model"] == selected]
    if not rowset.empty:
        row = rowset.iloc[0]
        matrix = [[int(row["validation_tn_at_best_f1"]), int(row["validation_fp_at_best_f1"])], [int(row["validation_fn_at_best_f1"]), int(row["validation_tp_at_best_f1"])]]
        col3, col4 = st.columns(2, gap="large")
        with col3:
            st.plotly_chart(
                confusion_chart(matrix, f"Validation confusion matrix · threshold {row['validation_best_f1_threshold']:.1%}", dark=dark),
                width="stretch",
                config={"displayModeBar": False},
            )
        with col4:
            thresholds = reports.get("thresholds")
            if isinstance(thresholds, pd.DataFrame) and not thresholds.empty:
                st.plotly_chart(threshold_tradeoff_chart(thresholds, dark=dark), width="stretch", config={"displayModeBar": False})
            else:
                st.info("Validation threshold sweep is unavailable.")

    st.markdown("#### Model comparison table")
    display_columns = {
        "model": "Model",
        "cv_mean_pr_auc": "CV PR-AUC mean",
        "cv_std_pr_auc": "CV PR-AUC SD",
        "validation_roc_auc": "Validation ROC-AUC",
        "validation_pr_auc": "Validation PR-AUC",
        "validation_precision_at_best_f1": "Validation precision",
        "validation_recall_at_best_f1": "Validation recall",
        "validation_best_f1": "Validation F1",
        "validation_best_f1_threshold": "Selected threshold",
    }
    available = [column for column in display_columns if column in comparison]
    table = comparison[available].rename(columns=display_columns).copy()
    if "Model" in table:
        table["Model"] = table["Model"].map({"xgboost": "XGBoost", "random_forest": "Random Forest", "logistic_regression": "Logistic Regression"}).fillna(table["Model"])
    table_style = table.style.set_properties(
        **{
            "background-color": "#1a293c" if dark else "#ffffff",
            "color": "#edf3fb" if dark else "#14283f",
            "border-color": "#34475f" if dark else "#d7e1ec",
        }
    )
    st.dataframe(table_style, width="stretch", hide_index=True)

    st.markdown("#### Preliminary test-partition diagnostics")
    st.warning("These saved test results are preliminary development diagnostics because the partition was evaluated multiple times. They are not pristine final holdout results.")
    test = reports.get("test")
    if test:
        selected_test = test.get("test_at_selected_threshold", {})
        st.caption(f"Selected validation threshold: {float(test.get('decision_threshold', float('nan'))):.4f}")
        cols = st.columns(5)
        for column, key, label in zip(cols, ("roc_auc", "pr_auc", "precision", "recall", "f1"), ("ROC-AUC", "PR-AUC", "Precision", "Recall", "F1")):
            value = selected_test.get(key)
            column.metric(label, f"{float(value):.3f}" if value is not None else "Unavailable")
        confusion = [
            [selected_test.get("tn"), selected_test.get("fp")],
            [selected_test.get("fn"), selected_test.get("tp")],
        ]
        if all(value is not None for row_values in confusion for value in row_values):
            st.plotly_chart(confusion_chart(confusion, "Preliminary test diagnostics · saved threshold", dark=dark), width="stretch", config={"displayModeBar": False})
    else:
        st.info("The saved preliminary test metrics file is unavailable.")
