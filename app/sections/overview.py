"""Portfolio overview section."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.charts import class_distribution_chart, top_feature_chart
from app.labels import MODEL_LABELS
from app.theme import metric_card, page_heading, panel_heading


def render(
    dataset_summary: dict[str, Any] | None,
    reports: dict[str, Any],
    health: dict[str, Any] | None,
) -> None:
    dark = st.session_state.get("appearance") == "Dark"
    page_heading(
        "Credit risk intelligence",
        "Lendwise helps explore loan outcomes, model behavior, and borrower risk estimates using the saved XGBoost pipeline.",
        eyebrow="LENDWISE · OVERVIEW",
        service_ready=bool(health and health.get("model_loaded")),
    )
    comparison = reports.get("comparison")
    test_report = reports.get("test")
    metadata = reports.get("metadata") or {}
    selected = str((test_report or {}).get("selected_model", metadata.get("selected_model", "xgboost")))
    threshold = (test_report or {}).get("decision_threshold", metadata.get("selected_threshold"))
    model_label = MODEL_LABELS.get(selected, selected.replace("_", " ").title())

    card_columns = st.columns(4, gap="medium")
    if dataset_summary:
        total = dataset_summary["rows"]
        defaults = dataset_summary["target_counts"].get(1, 0)
        default_rate = dataset_summary["target_percentages"].get(1, 0.0)
        with card_columns[0]:
            metric_card("Source rows", f"{total:,}", "Credit risk dataset")
        with card_columns[1]:
            metric_card("Default share", f"{default_rate:.1f}%", f"{defaults:,} observed defaults")
        with card_columns[2]:
            metric_card("Features", str(dataset_summary["features"]), "Saved model input schema")
    else:
        with card_columns[0]:
            metric_card("Source rows", "Unavailable", "Dataset file not found")
        with card_columns[1]:
            metric_card("Default share", "Unavailable", "No dataset summary")
        with card_columns[2]:
            metric_card("Features", "11", "Saved model input schema")
    with card_columns[3]:
        metric_card("Selected model", model_label, f"Threshold {float(threshold):.1%}" if threshold is not None else "Saved artifact")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.warning(
        "Evaluation status: current test-partition results are **preliminary development diagnostics**. "
        "The partition was evaluated multiple times and is not an untouched final holdout."
    )
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        panel_heading("Dataset profile", "Observed labels and basic data quality from the local CSV.")
        if dataset_summary:
            st.plotly_chart(class_distribution_chart(dataset_summary["target_counts"], dark=dark), width="stretch", config={"displayModeBar": False})
            st.caption(
                f"{dataset_summary['columns']} columns · {dataset_summary['missing_cells']:,} missing feature cells · "
                f"{dataset_summary['duplicate_rows']:,} exact duplicate rows."
            )
            with st.expander("Explore observed feature summaries"):
                numeric = pd.DataFrame(dataset_summary["numeric_summary"]).T
                numeric.index = [
                    {"person_age": "Borrower age", "person_income": "Annual income", "person_emp_length": "Employment length", "loan_amnt": "Loan amount", "loan_int_rate": "Interest rate", "loan_percent_income": "Loan-to-income ratio", "cb_person_cred_hist_length": "Credit history length"}.get(name, name)
                    for name in numeric.index
                ]
                st.markdown("**Numerical features**")
                numeric_style = numeric.round(2).style.set_properties(
                    **{
                        "background-color": "#1a293c" if dark else "#ffffff",
                        "color": "#edf3fb" if dark else "#14283f",
                        "border-color": "#34475f" if dark else "#d7e1ec",
                    }
                )
                st.dataframe(numeric_style, width="stretch")
                st.markdown("**Observed categorical values**")
                st.json(dataset_summary["categorical_values"], expanded=False)
        else:
            st.info("Dataset summary is unavailable. Add the compatible CSV at the configured local path to show observed class counts.")
    with right:
        panel_heading("What moves model predictions", "Exploratory SHAP attribution summary for the saved model.")
        importance = reports.get("importance")
        if isinstance(importance, pd.DataFrame) and not importance.empty:
            st.plotly_chart(top_feature_chart(importance, dark=dark), width="stretch", config={"displayModeBar": False})
            st.caption("Attributions describe fitted model behavior on a development training sample; they are not causal effects.")
        else:
            st.info("Exploratory feature importance report is unavailable.")

    if isinstance(comparison, pd.DataFrame) and not comparison.empty:
        selected_row = comparison.loc[comparison["model"] == selected]
        if not selected_row.empty:
            row = selected_row.iloc[0]
            st.markdown("### Selected model · validation snapshot")
            kpi = st.columns(3, gap="medium")
            with kpi[0]:
                metric_card("Training CV PR-AUC", f"{row['cv_mean_pr_auc']:.3f}", "Grouped CV mean")
            with kpi[1]:
                metric_card("Validation PR-AUC", f"{row['validation_pr_auc']:.3f}", "Model selection metric")
            with kpi[2]:
                metric_card("Validation ROC-AUC", f"{row['validation_roc_auc']:.3f}", "Validation partition")
    st.markdown("### Continue exploring")
    st.caption("Choose a workspace to review an example borrower, inspect evaluation reports, or explore model attributions.")
    targets = [
        ("Assess a borrower", "Risk assessment"),
        ("Review model analytics", "Model analytics"),
        ("Explore explainability", "Explainability"),
    ]
    nav_cols = st.columns(3, gap="medium")
    for column, (label, target) in zip(nav_cols, targets):
        column.button(
            label,
            use_container_width=True,
            key=f"overview_nav_{target}",
            on_click=_queue_navigation,
            args=(target,),
        )
    st.caption("Thresholds and probabilities support model review; they do not replace lending policy, affordability checks, or human oversight.")


def _queue_navigation(target: str) -> None:
    """Store navigation for the next run, before the sidebar radio is created."""
    st.session_state["_pending_section"] = target
