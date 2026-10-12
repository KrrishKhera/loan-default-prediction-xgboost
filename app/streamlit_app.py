"""Lendwise portfolio dashboard backed by the existing FastAPI service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from app.api_client import get_health
from app.data_service import read_csv_report, read_json_report, summarize_dataset
from app.sections import analytics, assessment, explainability, overview
from app.theme import brand, inject_styles

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(os.getenv("LOAN_DATASET_PATH", ROOT / "data/raw/credit_risk_dataset.csv"))
REPORTS_PATH = Path(os.getenv("LOAN_REPORTS_DIR", ROOT / "reports"))
API_URL = os.getenv("LOAN_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


@st.cache_data(show_spinner=False)
def _dataset_summary(path: str, modified_at: float | None) -> dict[str, Any] | None:
    del modified_at  # cache key changes when the underlying file timestamp changes
    try:
        frame = pd.read_csv(path)
        return summarize_dataset(frame)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def _reports(directory: str, modified: tuple[tuple[str, float | None], ...]) -> dict[str, Any]:
    del modified
    root = Path(directory)
    comparison = read_csv_report(root / "model_comparison.csv")
    thresholds = read_csv_report(root / "threshold_analysis.csv")
    importance = read_csv_report(root / "exploratory_shap/global_feature_importance.csv")

    def with_columns(frame: pd.DataFrame | None, columns: set[str]) -> pd.DataFrame | None:
        return frame if frame is not None and columns.issubset(frame.columns) else None

    return {
        "comparison": with_columns(
            comparison,
            {
                "model", "cv_mean_pr_auc", "cv_std_pr_auc", "validation_roc_auc",
                "validation_pr_auc", "validation_precision_at_best_f1",
                "validation_recall_at_best_f1", "validation_best_f1",
                "validation_best_f1_threshold", "validation_tn_at_best_f1",
                "validation_fp_at_best_f1", "validation_fn_at_best_f1",
                "validation_tp_at_best_f1",
            },
        ),
        "thresholds": with_columns(
            thresholds, {"threshold", "precision", "recall", "f1"}
        ),
        "test": read_json_report(root / "test_metrics.json"),
        "metadata": read_json_report(root / "run_metadata.json"),
        "importance": with_columns(
            importance, {"feature", "mean_abs_shap_probability"}
        ),
    }


def _file_signature(paths: list[Path]) -> tuple[tuple[str, float | None], ...]:
    signature = []
    for path in paths:
        try:
            signature.append((str(path), path.stat().st_mtime))
        except OSError:
            signature.append((str(path), None))
    return tuple(signature)


def _load_dashboard_data() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    try:
        modified = DATASET_PATH.stat().st_mtime
    except OSError:
        modified = None
    dataset = _dataset_summary(str(DATASET_PATH), modified)
    report_files = [
        REPORTS_PATH / "model_comparison.csv",
        REPORTS_PATH / "threshold_analysis.csv",
        REPORTS_PATH / "test_metrics.json",
        REPORTS_PATH / "exploratory_shap/global_feature_importance.csv",
    ]
    reports = _reports(str(REPORTS_PATH), _file_signature(report_files))
    return dataset, reports


def main() -> None:
    st.set_page_config(
        page_title="Lendwise · Credit Risk Intelligence",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    pending_section = st.session_state.pop("_pending_section", None)
    if pending_section in {"Overview", "Risk assessment", "Model analytics", "Explainability"}:
        # Apply the requested destination before the radio widget is instantiated.
        st.session_state["section"] = pending_section
    st.session_state.setdefault("appearance", "Light")
    inject_styles(st.session_state["appearance"])
    with st.sidebar:
        brand()
        st.markdown("<div class='eyebrow'>WORKSPACE</div>", unsafe_allow_html=True)
        section = st.radio(
            "Workspace navigation",
            ["Overview", "Risk assessment", "Model analytics", "Explainability"],
            label_visibility="collapsed",
            key="section",
        )
        st.divider()
        st.markdown("**Settings**")
        st.selectbox("Appearance", ["Light", "Dark"], key="appearance", help="Changes colors throughout the dashboard.")
        st.divider()
        st.caption("Decision support · model outputs require human review")
        st.caption("Connects to the configured risk API")

    dataset, reports = _load_dashboard_data()
    health = get_health(API_URL)

    if section == "Overview":
        overview.render(dataset, reports, health)
    elif section == "Risk assessment":
        assessment.render(API_URL, health)
    elif section == "Model analytics":
        analytics.render(reports, health)
    else:
        explainability.render(reports, health)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("Lendwise · Loan default prediction · Exploratory portfolio project · Existing test metrics remain preliminary.")


if __name__ == "__main__":
    main()
