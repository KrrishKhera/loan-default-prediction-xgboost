"""Global and individual exploratory SHAP presentation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.charts import individual_attribution_chart, top_feature_chart
from app.labels import FEATURE_LABELS
from app.theme import health_readiness, page_heading


def render(reports: dict[str, Any], health: dict[str, Any] | None) -> None:
    dark = st.session_state.get("appearance") == "Dark"
    page_heading(
        "Explainability",
        "Explore how the fitted model uses its inputs, globally and for a submitted borrower profile.",
        eyebrow="MODEL TRANSPARENCY",
        service_ready=health_readiness(health),
    )
    st.markdown(
        '<div class="callout"><b>Interpretation:</b> SHAP values attribute changes in this model\'s predicted default probability relative to its background sample. They do not show that a factor causes default.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Global feature importance")
    importance = reports.get("importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        st.plotly_chart(top_feature_chart(importance, limit=len(importance), dark=dark), width="stretch", config={"displayModeBar": False})
        st.caption("Exploratory summary from 300 development training-partition rows and an 80-row SHAP background. A larger mean absolute attribution means the feature moved this fitted model's outputs farther on average in this sample.")
    else:
        st.info("Global SHAP importance is unavailable because its saved report could not be read.")

    st.markdown("### Individual prediction explanation")
    result = st.session_state.get("last_explanation")
    if not result:
        st.info("To generate an individual explanation, open **Loan risk assessment** and choose **Assess + explain this case**. The request uses the existing FastAPI `/explain` endpoint.")
        return
    st.markdown(
        f"**{result['default_probability']:.1%} predicted default probability** · "
        f"{result['risk_label'].replace('_', ' ').title()} · threshold {result['decision_threshold']:.1%}"
    )
    st.caption(
        f"Background expectation: {result['base_probability']:.1%} · maximum additivity error: "
        f"{result['additivity_error']:.2e}. Contributions sum to the model probability within numeric tolerance."
    )
    attributions = result.get("feature_attributions", [])
    if attributions:
        st.plotly_chart(individual_attribution_chart(attributions, dark=dark), width="stretch", config={"displayModeBar": False})
        readable = pd.DataFrame(attributions)
        readable["Feature"] = readable["feature"].map(FEATURE_LABELS).fillna(readable["feature"])
        readable["Submitted value"] = readable["input_value"]
        readable["Probability contribution"] = readable["shap_value_probability"].map(lambda value: f"{value:+.2%}")
        readable["Effect in this model"] = readable["direction"].map({"raises": "Raises", "lowers": "Lowers", "no material change": "No material change"})
        readable_style = readable[
            ["Feature", "Submitted value", "Probability contribution", "Effect in this model"]
        ].style.set_properties(
            **{
                "background-color": "#1a293c" if dark else "#ffffff",
                "color": "#edf3fb" if dark else "#14283f",
                "border-color": "#34475f" if dark else "#d7e1ec",
            }
        )
        st.dataframe(readable_style, width="stretch", hide_index=True)
    st.caption("Correlated features can share attribution, and explanations depend on the background sample and perturbation assumptions. Treat this as model debugging context rather than causal evidence.")
