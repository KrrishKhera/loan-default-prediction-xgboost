"""Streamlit visual system and reusable layout primitives."""

from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st


THEMES = {
    "Light": {"ink": "#14283f", "muted": "#4e6279", "blue": "#2457c5", "line": "#d7e1ec", "surface": "#ffffff", "canvas": "#f3f6fa", "panel": "#f7faff", "control": "#ffffff", "shadow": "rgba(20,48,79,.06)"},
    "Dark": {"ink": "#edf3fb", "muted": "#bac8d9", "blue": "#82adff", "line": "#34475f", "surface": "#1a293c", "canvas": "#101a29", "panel": "#203249", "control": "#142235", "shadow": "rgba(0,0,0,.22)"},
}


def inject_styles(appearance: str = "Light") -> None:
    colors = THEMES.get(appearance, THEMES["Light"])
    st.markdown(
        f"""
        <style>
        :root {{ --ink:{colors['ink']}; --muted:{colors['muted']}; --blue:{colors['blue']}; --line:{colors['line']}; --surface:{colors['surface']}; --canvas:{colors['canvas']}; --panel:{colors['panel']}; --control:{colors['control']}; --shadow:{colors['shadow']}; }}
        html, body, [class*="css"] {{ font-family:Inter,'Avenir Next',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;color:var(--ink); }}
        [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:var(--canvas);color:var(--ink); }}
        [data-testid="stHeader"] {{ background:transparent; }}
        [data-testid="stMainBlockContainer"] {{ width:100%;max-width:1760px;padding:clamp(1.25rem,2.4vw,2.7rem) clamp(1rem,3.2vw,3.5rem) 3.5rem; }}
        [data-testid="stSidebar"] {{ background:var(--surface);border-right:1px solid var(--line); }}
        [data-testid="stSidebarCollapseButton"] {{ box-sizing:border-box;display:flex;align-items:center;justify-content:center;flex:0 0 2.5rem;width:2.5rem;min-width:2.5rem;height:2.5rem;margin-left:auto;overflow:visible;border-radius:10px;visibility:visible !important; }}
        [data-testid="stSidebarCollapseButton"] button, button[data-testid="stExpandSidebarButton"] {{ box-sizing:border-box;background-color:var(--surface) !important;border:1px solid var(--line) !important;border-radius:10px;color:var(--blue) !important;transition:background-color .15s ease,border-color .15s ease,color .15s ease; }}
        [data-testid="stSidebarCollapseButton"] button:hover, button[data-testid="stExpandSidebarButton"]:hover {{ background-color:var(--panel) !important;border-color:var(--blue) !important;color:var(--blue) !important; }}
        [data-testid="stSidebarCollapseButton"] button:focus-visible, button[data-testid="stExpandSidebarButton"]:focus-visible {{ border-color:var(--blue) !important;outline:3px solid var(--blue) !important;outline-offset:2px; }}
        [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"], button[data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{ color:var(--blue) !important; }}
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ color:var(--ink); }}
        [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p, .stCaption, .stCaption p {{ color:var(--muted) !important;opacity:1 !important; }}
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] span {{ color:var(--ink) !important;opacity:1 !important;font-weight:600; }}
        [data-testid="stSidebar"] [data-testid="stRadio"] label, [data-testid="stSidebar"] [data-testid="stRadio"] p {{ color:var(--ink) !important;opacity:1 !important;font-weight:600; }}
        [data-testid="stSidebar"] [data-testid="stRadio"] [aria-checked="true"] {{ color:var(--blue) !important; }}
        input, textarea, [data-baseweb="select"] *, [data-baseweb="input"] *, [data-baseweb="textarea"] * {{ color:var(--ink) !important; }}
        input, textarea, [data-baseweb="select"] > div, [data-baseweb="input"] > div {{ background:var(--control) !important;border-color:var(--line) !important; }}
        input::placeholder, textarea::placeholder {{ color:var(--muted) !important;opacity:1 !important; }}
        [role="listbox"], [role="option"] {{ background:var(--surface) !important;color:var(--ink) !important; }}
        h1,h2,h3,h4 {{ color:var(--ink) !important;font-family:Inter,'Avenir Next',-apple-system,sans-serif;letter-spacing:-.03em; }}
        h1 {{ font-size:clamp(1.8rem,2.3vw,2.35rem) !important;font-weight:800 !important; }}
        h2 {{ font-size:1.42rem !important;font-weight:750 !important; }}
        h3 {{ font-size:1.08rem !important;font-weight:750 !important; }}
        p, label, li {{ line-height:1.55; }}
        .brand-mark {{ display:flex; align-items:center; gap:.7rem; margin:.2rem 0 1.4rem; }}
        .brand-icon {{ background:#173a77;color:#fff;width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font:800 14px Inter,sans-serif;box-shadow:0 4px 10px #183d9129; }}
        .brand-name {{ color:var(--ink);font:800 16px Inter,sans-serif;line-height:1.1; }}
        .brand-caption {{ color:var(--muted);font-size:12px;margin-top:4px; }}
        .eyebrow {{ color:var(--blue);font-size:.73rem;text-transform:uppercase;letter-spacing:.13em;font-weight:800;margin-bottom:.4rem; }}
        .page-subtitle {{ color:var(--muted);font-size:1rem;margin-top:-.55rem;margin-bottom:1.55rem; }}
        .topline {{ display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:.7rem; }}
        .status-pill {{ display:inline-flex;align-items:center;gap:.45rem;padding:.38rem .72rem;border:1px solid {('#bfdfd8' if appearance == 'Light' else '#32675d')};border-radius:999px;background:{('#effaf6' if appearance == 'Light' else '#173b39')};color:{('#176956' if appearance == 'Light' else '#a7e6d4')};font-size:.75rem;font-weight:700;white-space:nowrap; }}
        .status-dot {{ width:7px;height:7px;border-radius:50%;background:#12a37e; }}
        .status-pill.delayed {{ border-color:{('#e7d9bd' if appearance == 'Light' else '#66562f')};background:{('#fffaf0' if appearance == 'Light' else '#41371f')};color:{('#805d1a' if appearance == 'Light' else '#f1d38b')}; }}
        .status-pill.delayed .status-dot {{ background:#bb8825; }}
        .status-pill:not(.online):not(.offline) {{ border-color:{('#e7d9bd' if appearance == 'Light' else '#66562f')};background:{('#fffaf0' if appearance == 'Light' else '#41371f')};color:{('#805d1a' if appearance == 'Light' else '#f1d38b')}; }}
        .status-pill:not(.online):not(.offline) .status-dot {{ background:#bb8825; }}
        .metric-card {{ height:100%;min-height:128px;background:var(--surface);border:1px solid var(--line);border-radius:15px;padding:17px 18px;box-shadow:0 3px 12px var(--shadow); }}
        .metric-label {{ color:var(--muted);font-size:.8rem;font-weight:700; }}
        .metric-value {{ color:var(--ink);font:800 1.7rem Inter,sans-serif;margin-top:.55rem;letter-spacing:-.04em; }}
        .metric-note {{ color:var(--muted);font-size:.76rem;margin-top:.38rem; }}
        .panel {{ background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:19px 20px;box-shadow:0 3px 14px var(--shadow); }}
        .callout {{ border:1px solid var(--line);background:var(--panel);border-radius:13px;padding:14px 16px;color:var(--ink);font-size:.9rem; }}
        .risk-card {{ border:1px solid var(--line);border-radius:17px;background:var(--surface);padding:22px 23px;box-shadow:0 6px 18px var(--shadow); }}
        .risk-label {{ font-size:.78rem;font-weight:800;letter-spacing:.1em;text-transform:uppercase; }}
        .risk-prob {{ font:800 2.7rem Inter,sans-serif;letter-spacing:-.05em;color:var(--ink);margin:.3rem 0; }}
        .risk-low {{ color:{'#08664f' if appearance == 'Light' else '#73dfbd'}; }}
        .risk-high {{ color:{'#9c421a' if appearance == 'Light' else '#ffb17b'}; }}
        .risk-detail {{ color:var(--muted);font-size:.86rem; }}
        [data-testid="stForm"] [data-testid="stFormSubmitButton"] button, .st-key-retry_service_connection [data-testid="stButton"] button {{ color:#FFFFFF !important; }}
        [data-testid="stForm"] [data-testid="stFormSubmitButton"] button *, .st-key-retry_service_connection [data-testid="stButton"] button * {{ color:#FFFFFF !important; }}
        .st-key-overview_navigation [data-testid="stButton"] button {{ display:flex;align-items:center;justify-content:center;min-height:3.25rem;padding:.72rem 1rem;border:1px solid var(--blue) !important;border-radius:10px;background:var(--blue) !important;color:var(--canvas) !important;font-weight:750;line-height:1.35;text-align:center;white-space:normal;transition:filter .15s ease,box-shadow .15s ease; }}
        .st-key-overview_navigation [data-testid="stButton"] button * {{ color:var(--canvas) !important; }}
        .st-key-overview_navigation [data-testid="stButton"] button:hover {{ filter:brightness(.92); }}
        .st-key-overview_navigation [data-testid="stButton"] button:focus-visible {{ outline:3px solid var(--blue) !important;outline-offset:3px; }}
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {{ background:#1e4da8;color:white;border:1px solid #1e4da8;border-radius:10px;font-weight:700;padding:.58rem 1rem;transition:background .15s ease; }}
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {{ background:#173d89;border-color:#173d89;color:white; }}
        div.stButton > button:focus, div[data-testid="stFormSubmitButton"] > button:focus {{ outline:3px solid #9eb9f2;outline-offset:2px; }}
        [data-testid="stMetric"] {{ background:var(--surface);border:1px solid var(--line);border-radius:13px;padding:13px 15px; }}
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {{ color:var(--muted) !important;font-weight:650; }}
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ color:var(--ink) !important; }}
        [data-testid="stPlotlyChart"] {{ background:transparent; }}
        [data-testid="stForm"] {{ background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:17px 18px 8px; }}
        [data-testid="stAlert"] {{ border-radius:12px;background:var(--panel) !important;border:1px solid var(--line); }}
        [data-testid="stAlert"] > div {{ background:var(--panel) !important; }}
        [data-testid="stAlert"] p, [data-testid="stAlert"] li {{ color:var(--ink) !important;opacity:1 !important; }}
        [data-testid="stDataFrame"] {{ border:1px solid var(--line);border-radius:12px;overflow:hidden; }}
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary span {{ color:var(--ink) !important;font-weight:650; }}
        hr {{ border-color:var(--line); }}
        @media (max-width:800px) {{
          [data-testid="stMainBlockContainer"] {{ padding:1.15rem .9rem 2.2rem; }}
          h1 {{ font-size:1.75rem !important; }}
          .topline {{ align-items:flex-start;flex-direction:column; }}
          .metric-card {{ min-height:108px;padding:14px; }}
          .metric-value {{ font-size:1.42rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def brand() -> None:
    st.markdown(
        '<div class="brand-mark"><div class="brand-icon">LR</div><div><div class="brand-name">Lendwise</div><div class="brand-caption">Credit risk intelligence</div></div></div>',
        unsafe_allow_html=True,
    )


def health_readiness(health: dict[str, Any] | None) -> bool | None:
    """Translate the API health payload into ready, delayed, or not-yet-checked."""
    if health is None:
        return None
    return health.get("status") == "ok" and health.get("model_loaded") is True


def page_heading(
    title: str,
    subtitle: str,
    *,
    eyebrow: str = "LENDWISE · CREDIT ANALYTICS",
    service_ready: bool | None = None,
) -> None:
    status_text = (
        "Prediction service ready"
        if service_ready is True
        else "Service starting · retry available"
        if service_ready is False
        else "Connecting to prediction service…"
    )
    status_class = "online" if service_ready is True else "delayed" if service_ready is False else "unknown"
    st.markdown(
        f'<div class="topline"><div><div class="eyebrow">{escape(eyebrow)}</div><h1>{escape(title)}</h1></div><div class="status-pill {status_class}"><span class="status-dot"></span>{escape(status_text)}</div></div>'
        f'<div class="page-subtitle">{escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        '<div class="metric-card">'
        f'<div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-note">{escape(note)}</div></div>',
        unsafe_allow_html=True,
    )


def panel_heading(title: str, caption: str | None = None) -> None:
    st.markdown(f"### {escape(title)}")
    if caption:
        st.caption(caption)


def risk_result_card(result: dict[str, Any]) -> None:
    probability = float(result["default_probability"])
    label = str(result["risk_label"])
    high = label == "higher_risk"
    css = "risk-high" if high else "risk-low"
    readable = "Higher risk" if high else "Lower risk"
    st.markdown(
        '<div class="risk-card">'
        f'<div class="risk-label {css}">{escape(readable)}</div>'
        f'<div class="risk-prob">{probability:.1%}</div>'
        '<div class="risk-detail">Predicted probability of default</div>'
        f'<hr><div class="risk-detail" style="display:flex;gap:2rem;flex-wrap:wrap">'
        f'<span><b>Decision threshold</b><br>{float(result["decision_threshold"]):.1%}</span>'
        f'<span><b>Predicted class</b><br>{"Default" if result["predicted_default"] else "Non-default"}</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )
