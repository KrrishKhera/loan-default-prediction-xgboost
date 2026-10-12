"""FastAPI serving layer for the saved loan-default pipeline."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from loan_default.data import FEATURES, load_data, split_data
from loan_default.explain import (
    build_shap_explainer,
    explain_instances,
    validate_artifact_schema,
)
from api.schemas import (
    BorrowerInput,
    ExplanationResponse,
    FeatureAttribution,
    HealthResponse,
    PredictionResponse,
)

logger = logging.getLogger(__name__)
DEFAULT_MODEL_PATH = Path("models/selected_model.joblib")
DEFAULT_BACKGROUND_PATH = Path("data/raw/credit_risk_dataset.csv")
RISK_LABEL_DEFINITION = (
    "higher_risk means predicted default probability is at or above the saved "
    "validation-selected threshold; lower_risk means it is below that threshold."
)


def _build_app(model_path: Path, background_path: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Model and SHAP explainer are initialized once per application process.
        app.state.artifact = None
        app.state.model_error = None
        app.state.explainer = None
        app.state.explanation_error = None
        try:
            artifact = joblib.load(model_path)
            validate_artifact_schema(artifact)
            threshold = float(artifact["decision_threshold"])
            if not 0 <= threshold <= 1:
                raise ValueError("Saved decision threshold is outside [0, 1]")
            app.state.artifact = artifact
        except Exception as exc:
            logger.exception("Unable to load saved model artifact")
            app.state.model_error = exc

        if app.state.artifact is not None:
            try:
                frame = load_data(background_path)
                training_rows = split_data(frame, random_state=42)["X_train"]
                assert isinstance(training_rows, pd.DataFrame)
                background = training_rows.sample(
                    n=min(80, len(training_rows)), random_state=42
                )
                app.state.explainer = build_shap_explainer(
                    app.state.artifact, background
                )
            except Exception as exc:
                # Predictions remain available if the local explanation
                # background is absent; explain requests receive a safe 503.
                logger.exception("Unable to initialize SHAP explainer")
                app.state.explanation_error = exc
        yield

    app = FastAPI(
        title="Loan Default Prediction API",
        description=(
            "Predicts default probability with the saved XGBoost pipeline. "
            "Risk labels apply the saved validation-selected threshold. "
            "Explanations are model attributions, not causal claims."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected API error for %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected server error occurred."},
        )

    def get_artifact(request: Request) -> dict[str, Any]:
        artifact = request.app.state.artifact
        if artifact is None:
            raise HTTPException(
                status_code=503,
                detail="Prediction model is unavailable. Check server configuration.",
            )
        return artifact

    def predict(artifact: dict[str, Any], borrower: BorrowerInput) -> tuple[float, float, bool]:
        values = borrower.model_dump()
        row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
        probability = float(artifact["pipeline"].predict_proba(row)[0, 1])
        threshold = float(artifact["decision_threshold"])
        return probability, threshold, probability >= threshold

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        model_ready = request.app.state.artifact is not None
        explanation_ready = request.app.state.explainer is not None
        return HealthResponse(
            status="ok" if model_ready else "degraded",
            model_loaded=model_ready,
            explanation_ready=explanation_ready,
        )

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    async def predict_endpoint(
        borrower: BorrowerInput, request: Request
    ) -> PredictionResponse:
        artifact = get_artifact(request)
        probability, threshold, is_default = predict(artifact, borrower)
        return PredictionResponse(
            default_probability=probability,
            decision_threshold=threshold,
            predicted_default=is_default,
            risk_label="higher_risk" if is_default else "lower_risk",
            risk_label_definition=RISK_LABEL_DEFINITION,
        )

    @app.post("/explain", response_model=ExplanationResponse, tags=["explanation"])
    async def explain_endpoint(
        borrower: BorrowerInput, request: Request
    ) -> ExplanationResponse:
        artifact = get_artifact(request)
        explainer = request.app.state.explainer
        if explainer is None:
            raise HTTPException(
                status_code=503,
                detail="Explanation service is unavailable; verify its background data.",
            )
        values = borrower.model_dump()
        row = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
        result = explain_instances(artifact, row, explainer)
        probability = float(result["probabilities"][0])
        threshold = float(artifact["decision_threshold"])
        is_default = probability >= threshold
        contributions = result["contributions"].iloc[0]
        attributions = []
        for feature in FEATURES:
            shap_value = float(contributions[feature])
            direction = (
                "raises" if shap_value > 1e-8 else "lowers" if shap_value < -1e-8 else "no material change"
            )
            attributions.append(
                FeatureAttribution(
                    feature=feature,
                    input_value=str(values[feature]),
                    shap_value_probability=shap_value,
                    direction=direction,
                )
            )
        return ExplanationResponse(
            default_probability=probability,
            decision_threshold=threshold,
            predicted_default=is_default,
            risk_label="higher_risk" if is_default else "lower_risk",
            risk_label_definition=RISK_LABEL_DEFINITION,
            base_probability=float(result["base_value"]),
            additivity_error=float(result["additivity_errors"][0]),
            explanation_method="SHAP TreeExplainer; probability scale; training-sample background",
            feature_attributions=attributions,
        )

    return app


def create_app(
    model_path: str | Path | None = None,
    background_path: str | Path | None = None,
) -> FastAPI:
    """Create an app; paths can be set with LOAN_MODEL_PATH/LOAN_BACKGROUND_PATH."""
    return _build_app(
        Path(model_path or os.getenv("LOAN_MODEL_PATH", DEFAULT_MODEL_PATH)),
        Path(background_path or os.getenv("LOAN_BACKGROUND_PATH", DEFAULT_BACKGROUND_PATH)),
    )


app = create_app()
