from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app
from loan_default.data import FEATURES

MODEL = "models/selected_model.joblib"
BACKGROUND = "data/raw/credit_risk_dataset.csv"


def valid_borrower() -> dict[str, object]:
    return {
        "person_age": 30.0,
        "person_income": 60000.0,
        "person_emp_length": 5.0,
        "loan_amnt": 10000.0,
        "loan_int_rate": 11.0,
        "loan_percent_income": 0.17,
        "cb_person_cred_hist_length": 7.0,
        "person_home_ownership": "RENT",
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "cb_person_default_on_file": "N",
    }


def test_health_and_valid_prediction_response():
    with TestClient(create_app(MODEL, BACKGROUND)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {
            "status": "ok",
            "model_loaded": True,
            "explanation_ready": True,
        }

        response = client.post("/predict", json=valid_borrower())
        assert response.status_code == 200
        body = response.json()
        assert 0 <= body["default_probability"] <= 1
        assert 0 <= body["decision_threshold"] <= 1
        assert body["risk_label"] in {"lower_risk", "higher_risk"}
        assert body["predicted_default"] == (
            body["default_probability"] >= body["decision_threshold"]
        )
        assert "threshold" in body["risk_label_definition"]


def test_rejects_missing_target_extra_fields_invalid_types_and_categories():
    with TestClient(create_app(MODEL, BACKGROUND)) as client:
        missing = valid_borrower()
        missing.pop("person_age")
        assert client.post("/predict", json=missing).status_code == 422

        for field, value in (("loan_status", 1), ("unexpected", "x")):
            invalid = valid_borrower()
            invalid[field] = value
            assert client.post("/predict", json=invalid).status_code == 422

        invalid_type = valid_borrower()
        invalid_type["person_income"] = "60000"
        assert client.post("/predict", json=invalid_type).status_code == 422

        invalid_category = valid_borrower()
        invalid_category["loan_grade"] = "Z"
        assert client.post("/predict", json=invalid_category).status_code == 422

        invalid_value = valid_borrower()
        invalid_value["loan_amnt"] = -1.0
        assert client.post("/predict", json=invalid_value).status_code == 422


def test_explanation_response_uses_complete_raw_feature_schema():
    with TestClient(create_app(MODEL, BACKGROUND)) as client:
        response = client.post("/explain", json=valid_borrower())
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(item["feature"] for item in body["feature_attributions"]) == set(FEATURES)
        assert "loan_status" not in {item["feature"] for item in body["feature_attributions"]}
        assert 0 <= body["default_probability"] <= 1
        assert body["additivity_error"] < 2e-4
        reconstructed = body["base_probability"] + sum(
            item["shap_value_probability"] for item in body["feature_attributions"]
        )
        assert abs(reconstructed - body["default_probability"]) < 2e-4


def test_missing_model_artifact_returns_safe_unavailable_response(tmp_path):
    missing_model = tmp_path / "missing.joblib"
    with TestClient(create_app(missing_model, BACKGROUND)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is False
        response = client.post("/predict", json=valid_borrower())
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]


def test_missing_explanation_background_leaves_prediction_available(tmp_path):
    missing_background = tmp_path / "missing.csv"
    with TestClient(create_app(MODEL, missing_background)) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert client.get("/health").json()["explanation_ready"] is False
        assert client.post("/predict", json=valid_borrower()).status_code == 200
        assert client.post("/explain", json=valid_borrower()).status_code == 503


def test_unexpected_prediction_error_returns_generic_server_error():
    class BrokenPipeline:
        def predict_proba(self, rows):
            raise RuntimeError("internal detail must not be exposed")

    with TestClient(create_app(MODEL, BACKGROUND), raise_server_exceptions=False) as client:
        client.app.state.artifact = {
            "pipeline": BrokenPipeline(),
            "decision_threshold": 0.5,
        }
        response = client.post("/predict", json=valid_borrower())
        assert response.status_code == 500
        assert response.json() == {"detail": "An unexpected server error occurred."}
        assert "internal detail" not in response.text
