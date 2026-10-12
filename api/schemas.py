"""Request and response contracts for the loan risk API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class BorrowerInput(BaseModel):
    """One borrower/loan record matching the saved model's raw feature schema."""

    model_config = ConfigDict(extra="forbid", strict=True)

    person_age: float = Field(gt=0)
    person_income: float = Field(ge=0)
    person_emp_length: float | None = Field(default=None, ge=0)
    loan_amnt: float = Field(gt=0)
    loan_int_rate: float | None = Field(default=None, ge=0)
    loan_percent_income: float = Field(ge=0)
    cb_person_cred_hist_length: float = Field(ge=0)
    person_home_ownership: Literal["MORTGAGE", "OTHER", "OWN", "RENT"]
    loan_intent: Literal[
        "DEBTCONSOLIDATION",
        "EDUCATION",
        "HOMEIMPROVEMENT",
        "MEDICAL",
        "PERSONAL",
        "VENTURE",
    ]
    loan_grade: Literal["A", "B", "C", "D", "E", "F", "G"]
    cb_person_default_on_file: Literal["N", "Y"]


class PredictionResponse(BaseModel):
    default_probability: float = Field(ge=0, le=1)
    decision_threshold: float = Field(ge=0, le=1)
    predicted_default: bool
    risk_label: Literal["lower_risk", "higher_risk"]
    risk_label_definition: str


class FeatureAttribution(BaseModel):
    feature: str
    input_value: str
    shap_value_probability: float
    direction: Literal["raises", "lowers", "no material change"]


class ExplanationResponse(PredictionResponse):
    base_probability: float = Field(ge=0, le=1)
    additivity_error: float = Field(ge=0)
    explanation_method: str
    feature_attributions: list[FeatureAttribution]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    explanation_ready: bool
