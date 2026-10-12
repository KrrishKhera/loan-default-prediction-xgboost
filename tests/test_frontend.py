from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import requests
from streamlit.testing.v1 import AppTest

from app.api_client import APIClientError, post_prediction
from app.charts import confusion_chart, top_feature_chart
from app.data_service import read_csv_report, read_json_report, summarize_dataset, validate_borrower_payload
from loan_default.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET
from app.theme import THEMES


def sample_dataset() -> pd.DataFrame:
    rows = []
    for i in range(4):
        row = {name: float(i + 1) for name in NUMERIC_FEATURES}
        row.update({name: f"category_{i % 2}" for name in CATEGORICAL_FEATURES})
        row[TARGET] = i % 2
        rows.append(row)
    return pd.DataFrame(rows)


def test_dataset_summary_reports_actual_counts_and_quality():
    frame = sample_dataset()
    frame.loc[0, "person_income"] = float("nan")
    summary = summarize_dataset(frame)
    assert summary["rows"] == 4
    assert summary["features"] == len(FEATURES)
    assert summary["missing_cells"] == 1
    assert summary["target_counts"] == {0: 2, 1: 2}
    assert summary["target_percentages"] == {0: 50.0, 1: 50.0}


def test_report_loaders_handle_missing_and_malformed_files(tmp_path):
    assert read_csv_report(tmp_path / "missing.csv") is None
    assert read_json_report(tmp_path / "missing.json") is None
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text('"unterminated')
    assert read_csv_report(bad_csv) is None
    good_json = tmp_path / "report.json"
    good_json.write_text(json.dumps({"metric": 0.8}))
    assert read_json_report(good_json) == {"metric": 0.8}


def test_borrower_payload_requires_exact_feature_schema_and_order():
    payload = {feature: index for index, feature in enumerate(reversed(FEATURES))}
    ordered = validate_borrower_payload(payload)
    assert list(ordered) == list(FEATURES)
    with pytest.raises(ValueError, match="unexpected"):
        validate_borrower_payload({**ordered, "loan_status": 1})
    with pytest.raises(ValueError, match="missing"):
        validate_borrower_payload({key: value for key, value in ordered.items() if key != FEATURES[0]})


def test_chart_theme_sets_explicit_legible_axis_tooltip_and_cell_colors():
    figure = confusion_chart([[5000, 42], [370, 1050]], "Validation matrix")
    assert figure.layout.font.color == "#20364e"
    assert figure.layout.xaxis.tickfont.color == "#304760"
    assert figure.layout.yaxis.tickfont.color == "#304760"
    assert figure.layout.hoverlabel.font.color == "#20364e"
    assert figure.data[0].colorscale[-1][1] == "#a7c0e8"


def test_dark_chart_palette_uses_readable_text_and_dark_surfaces():
    figure = confusion_chart([[5000, 42], [370, 1050]], "Validation matrix", dark=True)
    assert figure.layout.font.color == "#edf3fb"
    assert figure.layout.xaxis.tickfont.color == "#bac8d9"
    assert figure.layout.hoverlabel.bgcolor == "#1a293c"
    assert figure.layout.hoverlabel.font.color == "#edf3fb"
    assert figure.data[0].colorscale[-1][1] == "#526f98"
    assert figure.data[0].textfont.color == "#ffffff"


def test_theme_tokens_keep_surface_and_text_contrast_direction_consistent():
    assert THEMES["Light"]["ink"] == "#14283f"
    assert THEMES["Light"]["canvas"] == "#f3f6fa"
    assert THEMES["Dark"]["ink"] == "#edf3fb"
    assert THEMES["Dark"]["canvas"] == "#101a29"
    assert THEMES["Dark"]["surface"] != THEMES["Dark"]["canvas"]


def test_dark_global_shap_chart_uses_dark_theme_layout():
    importance = pd.DataFrame(
        {"feature": ["loan_amnt"], "mean_abs_shap_probability": [0.05]}
    )
    figure = top_feature_chart(importance, dark=True)
    assert figure.layout.font.color == "#edf3fb"
    assert figure.layout.yaxis.tickfont.color == "#bac8d9"
    assert figure.data[0].marker.color == "#82adff"


@pytest.mark.parametrize("appearance", ["Light", "Dark"])
def test_overview_navigation_buttons_switch_sections_once_and_preserve_theme(appearance):
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").run(timeout=30)
    assert not app.exception
    app.selectbox[0].set_value(appearance).run(timeout=30)
    assert not app.exception

    destinations = [
        ("overview_nav_Risk assessment", "Risk assessment"),
        ("overview_nav_Model analytics", "Model analytics"),
        ("overview_nav_Explainability", "Explainability"),
    ]
    for key, destination in destinations:
        app.radio[0].set_value("Overview").run(timeout=30)
        button = next(item for item in app.button if item.key == key)
        button.click().run(timeout=30)
        assert not app.exception
        assert app.radio[0].value == destination
        appearance_widget = next(item for item in app.selectbox if item.key == "appearance")
        assert appearance_widget.value == appearance


class StubResponse:
    def __init__(self, status_code: int, payload: object):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self.payload = payload

    def json(self):
        return self.payload


def test_api_client_returns_json_and_translates_safe_errors(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: StubResponse(200, {"default_probability": 0.2}))
    assert post_prediction("http://localhost:8000/", "/predict", {}) == {"default_probability": 0.2}

    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: StubResponse(422, {"detail": [{"loc": ["body", "loan_amnt"], "msg": "invalid"}]}),
    )
    with pytest.raises(APIClientError, match="loan_amnt"):
        post_prediction("http://localhost:8000", "/predict", {})

    def raise_timeout(*args, **kwargs):
        raise requests.Timeout("sensitive connection internals")

    monkeypatch.setattr(requests, "post", raise_timeout)
    with pytest.raises(APIClientError, match="took too long") as error:
        post_prediction("http://localhost:8000", "/predict", {})
    assert "sensitive" not in str(error.value)
