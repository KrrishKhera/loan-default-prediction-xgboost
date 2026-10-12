"""Borrower entry form and API-backed risk assessment."""

from __future__ import annotations

from typing import Any

import streamlit as st

from app.api_client import APIClientError, post_prediction
from app.charts import individual_attribution_chart
from app.data_service import validate_borrower_payload
from app.theme import page_heading, risk_result_card


def render(api_url: str, health: dict[str, Any] | None) -> None:
    dark = st.session_state.get("appearance") == "Dark"
    page_heading(
        "Loan risk assessment",
        "Enter the loan terms and borrower profile available at origination to estimate default probability.",
        eyebrow="BORROWER DECISION SUPPORT",
        service_ready=bool(health and health.get("model_loaded")),
    )
    st.markdown(
        '<div class="callout">The model uses the 11 fields below. Its output is an estimate for review, not an approved lending decision or a guarantee of actual repayment.</div>',
        unsafe_allow_html=True,
    )
    st.caption("The editable values shown in the form are illustrative starting values, not a recommended borrower profile.")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.form("borrower_assessment", clear_on_submit=False):
        st.markdown("#### Borrower profile")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Borrower age", min_value=1.0, max_value=120.0, value=30.0, step=1.0, help="Age in years.")
        income = c2.number_input("Annual income", min_value=0.0, value=60000.0, step=1000.0, help="Annual income in the dataset's currency units.")
        employment_provided = c3.checkbox("Include employment length", value=True, help="Turn off to let the model's fitted imputer handle this missing value.")
        emp_length = c3.number_input("Employment length (years)", min_value=0.0, value=5.0, step=1.0, disabled=not employment_provided)

        st.markdown("#### Loan terms")
        c4, c5, c6 = st.columns(3)
        amount = c4.number_input("Loan amount", min_value=1.0, value=10000.0, step=500.0, help="Requested/issued loan principal in the dataset's currency units.")
        rate_provided = c5.checkbox("Include interest rate", value=True, help="Turn off to let the model's fitted imputer handle this missing value.")
        interest = c5.number_input("Interest rate (%)", min_value=0.0, value=11.0, step=0.25, disabled=not rate_provided)
        ratio = c6.number_input("Loan-to-income ratio", min_value=0.0, value=0.17, step=0.01, format="%.3f", help="Loan amount divided by annual income, as represented in the source data.")

        st.markdown("#### Loan terms · continued")
        c7, c8 = st.columns(2)
        grade = c7.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"], index=1, help="Grade available at the chosen prediction point.")
        purpose = c8.selectbox("Loan purpose", ["DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"], index=1)

        st.markdown("#### Credit history")
        c9, c10, c11 = st.columns(3)
        history = c9.number_input("Credit history length (years)", min_value=0.0, value=7.0, step=1.0)
        home = c10.selectbox("Home ownership", ["MORTGAGE", "OTHER", "OWN", "RENT"], index=3)
        prior_default = c11.selectbox("Prior default on file", ["N", "Y"], index=0)

        payload = {
            "person_age": age,
            "person_income": income,
            "person_emp_length": emp_length if employment_provided else None,
            "loan_amnt": amount,
            "loan_int_rate": interest if rate_provided else None,
            "loan_percent_income": ratio,
            "cb_person_cred_hist_length": history,
            "person_home_ownership": home,
            "loan_intent": purpose,
            "loan_grade": grade,
            "cb_person_default_on_file": prior_default,
        }
        predict_button, explain_button = st.columns([1, 1])
        with predict_button:
            assess = st.form_submit_button("Run risk assessment", width="stretch")
        with explain_button:
            explain = st.form_submit_button("Assess + explain this case", width="stretch")

    explanation_error: str | None = None
    if assess or explain:
        st.session_state.pop("last_explanation", None)
        try:
            ordered_payload = validate_borrower_payload(payload)
        except ValueError:
            st.error("The borrower form is incomplete. Review the inputs and try again.")
            return
        try:
            with st.spinner("Scoring this borrower with the saved model…"):
                prediction = post_prediction(api_url, "/predict", ordered_payload)
            st.session_state["last_prediction"] = prediction
            if explain:
                with st.spinner("Preparing a model attribution for this case…"):
                    try:
                        explanation = post_prediction(api_url, "/explain", ordered_payload, timeout=30.0)
                        st.session_state["last_explanation"] = explanation
                    except APIClientError:
                        explanation_error = "The probability is ready, but this case's explanation is temporarily unavailable."
        except APIClientError as exc:
            st.error(str(exc))
            return

    result = st.session_state.get("last_prediction")
    if result:
        st.markdown("### Assessment result")
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            risk_result_card(result)
        with right:
            st.markdown("#### How to read this result")
            st.write(result.get("risk_label_definition", "The label compares probability with the saved decision threshold."))
            st.write("The probability is the model's estimate for the submitted profile. It is not a guarantee, a credit decision, or a causal assessment of the borrower.")
            st.caption("Current test metrics are preliminary development diagnostics; this interactive estimate does not establish independent model performance.")
        if explanation_error:
            st.warning(explanation_error)
        elif explain and st.session_state.get("last_explanation"):
            explanation = st.session_state["last_explanation"]
            st.markdown("#### Why the model moved this estimate")
            st.caption("SHAP contributions describe this model's attribution for this case; they do not establish causes.")
            attributions = explanation.get("feature_attributions", [])
            if attributions:
                st.plotly_chart(individual_attribution_chart(attributions, dark=dark), width="stretch", config={"displayModeBar": False})
            st.success("The individual attribution is also available in the Explainability section.")
