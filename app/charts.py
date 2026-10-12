"""Plotly chart builders used across dashboard sections."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.labels import FEATURE_LABELS, MODEL_LABELS

NAVY = "#173a77"
BLUE = "#3974d8"
TEAL = "#168c91"
AMBER = "#d3842d"
PALETTE = [NAVY, BLUE, TEAL, "#83a7e8"]


def _layout(fig: go.Figure, *, height: int = 340, dark: bool = False) -> go.Figure:
    ink, muted, grid, edge, hover_bg = (
        ("#edf3fb", "#bac8d9", "#34475f", "#52677f", "#1a293c")
        if dark else ("#20364e", "#304760", "#e5ebf2", "#b8c7d8", "#ffffff")
    )
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=height,
        margin=dict(l=15, r=15, t=45, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Avenir Next, sans-serif", color=ink, size=13),
        title_font=dict(family="Inter, Avenir Next, sans-serif", size=16, color=ink),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=ink, size=12)),
        hoverlabel=dict(bgcolor=hover_bg, bordercolor=edge, font=dict(color=ink, size=13)),
    )
    fig.update_xaxes(showgrid=False, linecolor=edge, zeroline=False, tickfont=dict(color=muted, size=12), title_font=dict(color=ink, size=12), automargin=True)
    fig.update_yaxes(gridcolor=grid, linecolor=edge, zeroline=False, tickfont=dict(color=muted, size=12), title_font=dict(color=ink, size=12), automargin=True)
    return fig


def class_distribution_chart(counts: dict[int, int], *, dark: bool = False) -> go.Figure:
    labels = {0: "Non-default", 1: "Default"}
    frame = pd.DataFrame(
        {"Outcome": [labels.get(key, str(key)) for key in counts], "Borrowers": list(counts.values())}
    )
    fig = px.pie(
        frame,
        values="Borrowers",
        names="Outcome",
        hole=0.68,
        color="Outcome",
        color_discrete_map={"Non-default": "#3974d8", "Default": "#d3842d"},
    )
    fig.update_traces(textposition="outside", textinfo="label+percent", textfont=dict(color="#edf3fb" if dark else "#1d344d", size=13), marker_line_color="#1a293c" if dark else "white", marker_line_width=3)
    fig.update_layout(showlegend=False, title="Observed loan outcomes")
    return _layout(fig, height=340, dark=dark)


def top_feature_chart(importance: pd.DataFrame, *, limit: int = 8, dark: bool = False) -> go.Figure:
    frame = importance.head(limit).copy().sort_values("mean_abs_shap_probability")
    frame["label"] = frame["feature"].map(FEATURE_LABELS).fillna(frame["feature"])
    fig = go.Figure(
        go.Bar(
            x=frame["mean_abs_shap_probability"],
            y=frame["label"],
            orientation="h",
            marker_color="#82adff" if dark else BLUE,
            hovertemplate="%{y}<br>Mean |SHAP|: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(title="Largest average model attributions", xaxis_title="Mean absolute SHAP value (probability scale)")
    return _layout(fig, height=360, dark=dark)


def cv_chart(comparison: pd.DataFrame, *, dark: bool = False) -> go.Figure:
    frame = comparison.copy()
    frame["Model"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    fig = go.Figure(
        go.Bar(
            x=frame["Model"],
            y=frame["cv_mean_pr_auc"],
            error_y=dict(type="data", array=frame["cv_std_pr_auc"], visible=True),
            marker_color=(["#a9c5ff", "#82adff", "#5bd0c1"] if dark else PALETTE)[: len(frame)],
            text=frame["cv_mean_pr_auc"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            hovertemplate="%{x}<br>CV PR-AUC: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(title="Training-only grouped cross-validation", yaxis_title="Mean PR-AUC ± standard deviation", yaxis_range=[0, 1.04])
    return _layout(fig, height=350, dark=dark)


def validation_ranking_chart(comparison: pd.DataFrame, *, dark: bool = False) -> go.Figure:
    frame = comparison.melt(
        id_vars="model",
        value_vars=["validation_roc_auc", "validation_pr_auc"],
        var_name="Metric",
        value_name="Score",
    )
    frame["Model"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    frame["Metric"] = frame["Metric"].map(
        {"validation_roc_auc": "ROC-AUC", "validation_pr_auc": "PR-AUC"}
    )
    fig = px.bar(
        frame,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text=frame["Score"].map(lambda value: f"{value:.3f}"),
        color_discrete_sequence=["#82adff", "#5bd0c1"] if dark else [BLUE, TEAL],
    )
    fig.update_traces(textposition="outside", hovertemplate="%{x}<br>%{legendgroup}: %{y:.4f}<extra></extra>")
    fig.update_layout(title="Validation ranking metrics", yaxis_title="Score", yaxis_range=[0, 1.08])
    return _layout(fig, height=360, dark=dark)


def validation_threshold_chart(comparison: pd.DataFrame, *, dark: bool = False) -> go.Figure:
    frame = comparison.melt(
        id_vars="model",
        value_vars=[
            "validation_precision_at_best_f1",
            "validation_recall_at_best_f1",
            "validation_best_f1",
        ],
        var_name="Metric",
        value_name="Score",
    )
    friendly = {
        "validation_precision_at_best_f1": "Precision",
        "validation_recall_at_best_f1": "Recall",
        "validation_best_f1": "F1",
    }
    frame["Metric"] = frame["Metric"].map(friendly)
    frame["Model"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    colors = ["#a9c5ff", "#82adff", "#5bd0c1"] if dark else [NAVY, BLUE, TEAL]
    fig = px.bar(frame, x="Model", y="Score", color="Metric", barmode="group", text_auto=".3f", color_discrete_sequence=colors)
    fig.update_layout(title="Validation metrics at each model's F1-selected threshold", yaxis_title="Score", yaxis_range=[0, 1.12])
    return _layout(fig, height=360, dark=dark)


def threshold_tradeoff_chart(thresholds: pd.DataFrame, *, dark: bool = False) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (
        ("precision", "Precision", "#a9c5ff" if dark else NAVY),
        ("recall", "Recall", "#82adff" if dark else BLUE),
        ("f1", "F1", "#5bd0c1" if dark else TEAL),
    ):
        fig.add_trace(go.Scatter(x=thresholds["threshold"], y=thresholds[column], mode="lines+markers", name=label, line=dict(color=color, width=2.5), marker=dict(size=6)))
    fig.update_layout(title="XGBoost decision-threshold trade-off · validation", xaxis_title="Decision threshold", yaxis_title="Score", yaxis_range=[0, 1.03], hovermode="x unified")
    return _layout(fig, height=350, dark=dark)


def confusion_chart(matrix: list[list[int]], title: str, *, dark: bool = False) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted non-default", "Predicted default"],
            y=["Actual non-default", "Actual default"],
            colorscale=[[0, "#253a54"], [1, "#526f98"]] if dark else [[0, "#edf3fc"], [1, "#a7c0e8"]],
            showscale=False,
            text=[[f"{value:,}" for value in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=17, color="#ffffff" if dark else "#17324f"),
            hovertemplate="%{y}<br>%{x}<br>Rows: %{z:,}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_side="bottom", yaxis_autorange="reversed")
    return _layout(fig, height=300, dark=dark)


def individual_attribution_chart(attributions: list[dict[str, Any]], *, dark: bool = False) -> go.Figure:
    frame = pd.DataFrame(attributions)
    frame["label"] = frame["feature"].map(FEATURE_LABELS).fillna(frame["feature"])
    frame["color"] = frame["shap_value_probability"].map(lambda value: "Raises probability" if value > 0 else "Lowers probability" if value < 0 else "No material change")
    frame = frame.sort_values("shap_value_probability", key=lambda values: values.abs())
    fig = px.bar(
        frame,
        x="shap_value_probability",
        y="label",
        color="color",
        orientation="h",
        color_discrete_map={"Raises probability": "#ffb17b" if dark else "#b85428", "Lowers probability": "#73dfbd" if dark else "#1b7d68", "No material change": "#aebdd0" if dark else "#7890a9"},
        hover_data={"input_value": True, "shap_value_probability": ":.4f", "label": False, "color": False},
    )
    fig.update_layout(title="How this case moves the model probability", xaxis_title="SHAP contribution to default probability", yaxis_title="")
    fig.add_vline(x=0, line_color="#bac8d9" if dark else "#62748a", line_width=1)
    return _layout(fig, height=440, dark=dark)
