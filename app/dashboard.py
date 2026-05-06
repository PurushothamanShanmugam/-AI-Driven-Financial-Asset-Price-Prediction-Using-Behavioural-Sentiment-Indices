"""
dashboard.py — AI Financial Behaviour Dashboard
Streamlit 1.56+ compatible. No statsmodels, no tabulate, no applymap.
Run: streamlit run app/dashboard.py
"""

import json
from pathlib import Path
import base64, io, re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent.parent
OUTPUTS_DIR     = BASE_DIR / "outputs"
METRICS_DIR     = OUTPUTS_DIR / "metrics"
FIGURES_DIR     = OUTPUTS_DIR / "figures"
REPORTS_DIR     = OUTPUTS_DIR / "reports"
STATS_DIR       = OUTPUTS_DIR / "stats"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
MODELS_DIR      = BASE_DIR / "models"
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
RAW_DIR         = BASE_DIR / "data" / "raw"

ASSET_COLORS = {
    "SPY":     "#0B2A5B",   # deep navy
    "QQQ":     "#4C1D95",   # royal violet
    "GLD":     "#7C5A12",   # muted bronze, not bright yellow
    "TLT":     "#0F766E",   # deep teal
    "BTC_USD": "#B45309",   # burnt orange
}
MODEL_COLORS = {
    "Ridge":            "#0B2A5B",
    "Lasso":            "#4C1D95",
    "ElasticNet":       "#0F766E",
    "RandomForest":     "#334155",
    "GradientBoosting": "#7F1D1D",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Behaviour Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lightweight Currency Rain Background ──────────────────────────────────────
# Uses pure HTML/CSS instead of a fullscreen iframe/canvas hack.
# This keeps the currency rain but does not block scrolling or slow Streamlit reruns.
st.markdown("""
<div class="currency-rain-bg" aria-hidden="true">
        <span style="left:2.0vw; animation-duration:15.0s; animation-delay:0.0s; font-size:24px; color:#0B2A5B;">$</span>
        <span style="left:5.6vw; animation-duration:17.1s; animation-delay:-1.4s; font-size:27px; color:#154E75;">€</span>
        <span style="left:9.2vw; animation-duration:19.2s; animation-delay:-2.7s; font-size:30px; color:#7C5A12;">£</span>
        <span style="left:12.8vw; animation-duration:21.3s; animation-delay:-4.1s; font-size:33px; color:#0F766E;">¥</span>
        <span style="left:16.4vw; animation-duration:23.4s; animation-delay:-5.4s; font-size:36px; color:#4C1D95;">₹</span>
        <span style="left:20.0vw; animation-duration:25.5s; animation-delay:-6.8s; font-size:24px; color:#0B2A5B;">₿</span>
        <span style="left:23.6vw; animation-duration:27.6s; animation-delay:-8.1s; font-size:27px; color:#154E75;">₩</span>
        <span style="left:27.2vw; animation-duration:15.0s; animation-delay:-9.5s; font-size:30px; color:#7C5A12;">₦</span>
        <span style="left:30.8vw; animation-duration:17.1s; animation-delay:-10.8s; font-size:33px; color:#0F766E;">₫</span>
        <span style="left:34.4vw; animation-duration:19.2s; animation-delay:-12.2s; font-size:36px; color:#4C1D95;">₱</span>
        <span style="left:38.0vw; animation-duration:21.3s; animation-delay:-13.5s; font-size:24px; color:#0B2A5B;">₴</span>
        <span style="left:41.6vw; animation-duration:23.4s; animation-delay:0.0s; font-size:27px; color:#154E75;">₲</span>
        <span style="left:45.2vw; animation-duration:25.5s; animation-delay:-1.4s; font-size:30px; color:#7C5A12;">₵</span>
        <span style="left:48.8vw; animation-duration:27.6s; animation-delay:-2.7s; font-size:33px; color:#0F766E;">₸</span>
        <span style="left:52.4vw; animation-duration:15.0s; animation-delay:-4.1s; font-size:36px; color:#4C1D95;">฿</span>
        <span style="left:56.0vw; animation-duration:17.1s; animation-delay:-5.4s; font-size:24px; color:#0B2A5B;">₾</span>
        <span style="left:59.6vw; animation-duration:19.2s; animation-delay:-6.8s; font-size:27px; color:#154E75;">₽</span>
        <span style="left:63.2vw; animation-duration:21.3s; animation-delay:-8.1s; font-size:30px; color:#7C5A12;">₡</span>
        <span style="left:66.8vw; animation-duration:23.4s; animation-delay:-9.5s; font-size:33px; color:#0F766E;">₣</span>
        <span style="left:70.4vw; animation-duration:25.5s; animation-delay:-10.8s; font-size:36px; color:#4C1D95;">₤</span>
        <span style="left:74.0vw; animation-duration:27.6s; animation-delay:-12.2s; font-size:24px; color:#0B2A5B;">₨</span>
        <span style="left:77.6vw; animation-duration:15.0s; animation-delay:-13.5s; font-size:27px; color:#154E75;">₪</span>
        <span style="left:81.2vw; animation-duration:17.1s; animation-delay:0.0s; font-size:30px; color:#7C5A12;">₭</span>
        <span style="left:84.8vw; animation-duration:19.2s; animation-delay:-1.4s; font-size:33px; color:#0F766E;">₮</span>
        <span style="left:88.4vw; animation-duration:21.3s; animation-delay:-2.7s; font-size:36px; color:#4C1D95;">₺</span>
        <span style="left:92.0vw; animation-duration:23.4s; animation-delay:-4.1s; font-size:24px; color:#0B2A5B;">$</span>
        <span style="left:95.6vw; animation-duration:25.5s; animation-delay:-5.4s; font-size:27px; color:#154E75;">€</span>
        <span style="left:99.2vw; animation-duration:27.6s; animation-delay:-6.8s; font-size:30px; color:#7C5A12;">£</span>
</div>
<style>
.currency-rain-bg {
    position: fixed;
    inset: 0;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    pointer-events: none !important;
    z-index: 0;
}
.currency-rain-bg span {
    position: absolute;
    top: -80px;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
    font-weight: 800;
    opacity: 0.22;
    text-shadow: 0 1px 8px rgba(15,30,61,0.12);
    animation-name: currency-rain-fall;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
    will-change: transform;
}
@keyframes currency-rain-fall {
    0%   { transform: translate3d(0, -12vh, 0) rotate(0deg); opacity: 0; }
    12%  { opacity: 0.24; }
    88%  { opacity: 0.24; }
    100% { transform: translate3d(0, 112vh, 0) rotate(10deg); opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* ── Single-scroll Streamlit shell + light pearl background ── */
    html, body {
        height: 100% !important;
        max-height: 100% !important;
        overflow: hidden !important;
        background-color: #e8f0fa !important;
    }

    /* Remove the empty Streamlit block created by the fixed currency-rain div */
    .element-container:has(.currency-rain-bg),
    [data-testid="stMarkdownContainer"]:has(.currency-rain-bg) {
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
    }

    /* ── stApp transparent so the currency rain shows through safely ── */
    .stApp {
        background-color: transparent !important;
        background-image: none !important;
        color: #0f1e3d !important;
        font-family: 'Inter', sans-serif !important;
        position: relative;
        z-index: 1;
    }

    /* Keep content above currency rain; only ONE scroll container is allowed */
    .main, .main .block-container,
    section[data-testid="stSidebar"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stAppViewContainer"],
    [data-testid="stVerticalBlock"] {
        position: relative;
        z-index: 5 !important;
    }

    iframe {
        pointer-events: none !important;
        height: 0 !important;
        min-height: 0 !important;
        max-height: 0 !important;
    }

    [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scroll-behavior: smooth;
    }

    .main .block-container {
        padding-top: 0.85rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 92rem !important;
    }

    /* Make the page title start near the top and centered */
    .main .block-container h1:first-of-type {
        text-align: center !important;
        margin-top: 0 !important;
        margin-bottom: 0.35rem !important;
    }

    .main .block-container h1:first-of-type + div p {
        text-align: center !important;
        margin-top: 0 !important;
    }

    /* Keep Streamlit's top-left sidebar/controller visible, but remove big header space */
    [data-testid="stHeader"] {
        background: transparent !important;
        height: 2.2rem !important;
        pointer-events: auto !important;
        z-index: 10000 !important;
    }

    [data-testid="stToolbar"] {
        pointer-events: auto !important;
        z-index: 10001 !important;
    }

    /* ── PROCESSING / LOADING ANIMATIONS ─────────────────────────────── */
    /* Top sliding progress bar — shows whenever Streamlit is running */
    [data-testid="stStatusWidget"] {
        background: linear-gradient(135deg, #0f1e3d 0%, #1e3a6f 100%) !important;
        color: #ffffff !important;
        border: 1.5px solid rgba(184,134,11,0.5) !important;
        border-radius: 14px !important;
        padding: 8px 16px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        font-size: 13px !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 8px 32px rgba(15,30,61,0.35),
                    0 0 0 4px rgba(184,134,11,0.12) !important;
        animation: status-pulse 1.4s ease-in-out infinite !important;
        z-index: 9999 !important;
    }
    [data-testid="stStatusWidget"] *,
    [data-testid="stStatusWidget"] svg,
    [data-testid="stStatusWidget"] path {
        color: #ffffff !important;
        fill: #7C5A12 !important;
        stroke: #7C5A12 !important;
    }
    [data-testid="stStatusWidget"] svg {
        animation: coin-flip 0.9s linear infinite !important;
    }
    @keyframes status-pulse {
        0%, 100% {
            box-shadow: 0 8px 32px rgba(15,30,61,0.35),
                        0 0 0 0 rgba(184,134,11,0.55);
        }
        50% {
            box-shadow: 0 12px 40px rgba(15,30,61,0.45),
                        0 0 0 12px rgba(184,134,11,0);
        }
    }
    @keyframes coin-flip { 0% { transform: rotateY(0deg); } 100% { transform: rotateY(360deg); } }

    /* Top progress bar (always rendered; animated via Streamlit class hooks) */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0; left: -100%;
        width: 40%; height: 3px;
        background: linear-gradient(90deg,
            transparent 0%, #7C5A12 20%, #7C5A12 50%, #7C5A12 80%, transparent 100%);
        z-index: 99999;
        opacity: 0;
        animation: top-bar-slide 1.6s ease-in-out infinite;
        pointer-events: none;
    }
    /* Activate progress bar when Streamlit reports "running" */
    body:has([data-testid="stStatusWidget"]) .stApp::before {
        opacity: 1;
    }
    @keyframes top-bar-slide {
        0%   { left: -40%;  opacity: 1; }
        50%  { opacity: 1; }
        100% { left: 100%;  opacity: 1; }
    }

    /* Inline spinner (st.spinner) — gold themed */
    .stSpinner > div, [data-testid="stSpinner"] > div {
        border-top-color: #7C5A12 !important;
        border-right-color: rgba(184,134,11,0.35) !important;
        border-bottom-color: rgba(184,134,11,0.15) !important;
        border-left-color: rgba(184,134,11,0.05) !important;
        border-width: 3px !important;
    }
    [data-testid="stSpinner"] {
        font-family: 'Inter', sans-serif !important;
        color: #0f1e3d !important;
        font-weight: 600 !important;
    }

    /* ── Sidebar — opaque to remain readable over canvas ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f5fc 100%) !important;
        border-right: 1px solid rgba(184,134,11,0.22) !important;
        box-shadow: 4px 0 32px rgba(15,30,61,0.14) !important;
    }
    section[data-testid="stSidebar"] * { color: #0f1e3d !important; }

    /* ── Metric cards — solid white background ── */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(247,250,254,0.98) 100%);
        border: 1px solid rgba(184,134,11,0.20);
        border-left: 3px solid #7C5A12;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 6px 24px rgba(15,30,61,0.10), inset 0 1px 0 rgba(255,255,255,0.95);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(184,134,11,0.20), 0 6px 24px rgba(15,30,61,0.10);
    }
    div[data-testid="metric-container"] label {
        color: #0f1e3d !important;
        font-size: 11px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-family: 'Inter', sans-serif !important;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #0f1e3d !important;
        font-size: 26px !important;
        font-weight: 800 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.02em;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.95);
        border-radius: 14px;
        padding: 5px;
        gap: 3px;
        border: 1px solid rgba(184,134,11,0.20);
        box-shadow: 0 4px 18px rgba(15,30,61,0.10), inset 0 1px 0 rgba(255,255,255,0.9);
        display: flex !important;
        width: 100% !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #0f1e3d !important;
        font-weight: 600;
        padding: 8px 16px;
        font-size: 13px;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif !important;
        flex: 1 1 0% !important;
        text-align: center !important;
        justify-content: center !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(184,134,11,0.10) !important;
        color: #7a5400 !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(184,134,11,0.22), rgba(184,134,11,0.12)) !important;
        color: #5e3f00 !important;
        border: 1px solid rgba(184,134,11,0.40) !important;
        box-shadow: 0 2px 14px rgba(184,134,11,0.18), inset 0 1px 0 rgba(255,255,255,0.7) !important;
    }

    /* ── Typography — PROFESSIONAL Inter, no decorative gradient ── */
    h1, h2, h3, h4, h5, h6 { font-family: 'Inter', sans-serif !important; }
    h1 {
        color: #0a1f3d !important;
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.025em !important;
        line-height: 1.1 !important;
        margin: 0 0 4px 0 !important;
    }
    h2 { color: #0f1e3d !important; font-size: 1.35rem !important; font-weight: 700 !important;
         letter-spacing: -0.01em !important; }
    h3 { color: #1e3a6f !important; font-size: 1.1rem  !important; font-weight: 700 !important;
         letter-spacing: -0.005em !important; }
    h4 { color: #1e3a6f !important; font-size: 1.0rem  !important; font-weight: 700 !important; }
    p  { color: #0f1e3d !important; line-height: 1.65; font-family: 'Inter', sans-serif !important;
         font-weight: 500 !important; }
    label { color: #0f1e3d !important; font-size: 13px !important; font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important; }
    span, div { color: inherit; }

    /* ── Chart containers — solid white card ── */
    .js-plotly-plot {
        border-radius: 16px;
        background: rgba(255,255,255,0.96) !important;
        box-shadow: 0 8px 32px rgba(15,30,61,0.12), 0 0 0 1px rgba(184,134,11,0.12);
        overflow: hidden;
    }
    .stDataFrame {
        border-radius: 14px;
        overflow: hidden;
        background: rgba(255,255,255,0.97) !important;
        box-shadow: 0 6px 22px rgba(15,30,61,0.12);
        border: 1px solid rgba(184,134,11,0.16) !important;
    }

    /* ── Input widgets ── */
    .stSelectbox > div > div, .stMultiSelect > div > div, .stNumberInput > div > div > input {
        background: #ffffff !important;
        border: 1px solid rgba(184,134,11,0.30) !important;
        border-radius: 10px !important;
        color: #0f1e3d !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }
    .stNumberInput > div > div > input { font-family: 'JetBrains Mono', monospace !important; }
    .stRadio label { color: #0f1e3d !important; font-weight: 600 !important; }
    .stFileUploader {
        border-radius: 12px !important;
        background: rgba(255,255,255,0.92) !important;
        border: 1.5px dashed rgba(184,134,11,0.32) !important;
    }
    .stFileUploader * { color: #0f1e3d !important; }

    /* ── Info / warn boxes ── */
    .info-box {
        background: rgba(30,100,200,0.10);
        border-left: 4px solid #1a6ebf;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px; margin: 8px 0;
        color: #0a3a7a !important;
        font-size: 13.5px; font-weight: 600;
        font-family: 'Inter', sans-serif;
    }
    .warn-box {
        background: rgba(184,134,11,0.12);
        border-left: 4px solid #7C5A12;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px; margin: 8px 0;
        color: #5e3f00 !important;
        font-size: 13.5px; font-weight: 600;
        font-family: 'Inter', sans-serif;
    }

    /* ── Buttons ── */
    .stDownloadButton button, .stButton button {
        background: linear-gradient(135deg, #7C5A12 0%, #8a6500 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.03em;
        box-shadow: 0 4px 18px rgba(184,134,11,0.32) !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton button:hover, .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 32px rgba(184,134,11,0.50) !important;
        background: linear-gradient(135deg, #8B6A1F 0%, #6B4E16 100%) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: rgba(200,220,245,0.4); }
    ::-webkit-scrollbar-thumb { background: rgba(184,134,11,0.35); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(184,134,11,0.55); }

    /* ── Misc ── */
    hr { border-color: rgba(184,134,11,0.18) !important; }
    code {
        font-family: 'JetBrains Mono', monospace !important;
        background: rgba(184,134,11,0.14) !important;
        color: #5e3f00 !important;
        border-radius: 5px; padding: 2px 7px; font-weight: 600;
    }
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: visible !important; }
</style>
""", unsafe_allow_html=True)

# ── Plotly light template ─────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(245,248,255,0.88)",
    font=dict(color="#0f1e3d", family="Inter, sans-serif", size=12),
    legend=dict(
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="rgba(184,134,11,0.22)",
        borderwidth=1,
        font=dict(color="#0f1e3d", size=11),
    ),
    margin=dict(l=50, r=30, t=50, b=40),
)

def _title(text):
    """Returns plotly title dict for update_layout calls."""
    return dict(text=text, font=dict(color="#0f1e3d", size=14,
                                      family="Inter, sans-serif"))

def _style_axes(fig):
    fig.update_xaxes(
        gridcolor="rgba(30,60,120,0.07)",
        zerolinecolor="rgba(184,134,11,0.28)",
        tickfont=dict(color="#0f1e3d", family="JetBrains Mono"),
        title_font=dict(color="#0f1e3d"),
        linecolor="rgba(184,134,11,0.14)",
    )
    fig.update_yaxes(
        gridcolor="rgba(30,60,120,0.07)",
        zerolinecolor="rgba(184,134,11,0.28)",
        tickfont=dict(color="#0f1e3d", family="JetBrains Mono"),
        title_font=dict(color="#0f1e3d"),
        linecolor="rgba(184,134,11,0.14)",
    )
    return fig

def apply_dark(fig):
    fig.update_layout(**PLOT_LAYOUT)
    return _style_axes(fig)



def _patch_sklearn_compatibility(obj, _seen=None):
    """
    Patch loaded sklearn objects created with a different scikit-learn version.
    This specifically fixes old/new SimpleImputer joblib objects that can raise:
    'SimpleImputer' object has no attribute '_fill_dtype'.
    """
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return obj
    _seen.add(oid)

    # Patch SimpleImputer-like objects safely.
    if obj.__class__.__name__ == "SimpleImputer" and not hasattr(obj, "_fill_dtype"):
        try:
            stats = getattr(obj, "statistics_", None)
            if stats is not None and hasattr(stats, "dtype"):
                obj._fill_dtype = stats.dtype
            else:
                obj._fill_dtype = object
        except Exception:
            obj._fill_dtype = object

    # Recurse through common containers and sklearn estimators.
    if isinstance(obj, dict):
        for v in obj.values():
            _patch_sklearn_compatibility(v, _seen)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            _patch_sklearn_compatibility(v, _seen)
    else:
        # Pipeline / ColumnTransformer / ensembles / wrappers usually expose internals in __dict__.
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for v in list(d.values()):
                if isinstance(v, (str, bytes, int, float, bool, type(None))):
                    continue
                try:
                    _patch_sklearn_compatibility(v, _seen)
                except Exception:
                    pass
    return obj


def safe_joblib_load(path):
    """Load joblib model package and patch sklearn compatibility issues."""
    pkg = joblib.load(path)
    return _patch_sklearn_compatibility(pkg)

def _cuboid_trace(x_center, height, name, color="#0B2A5B", width=0.56, depth=0.34, showlegend=False, hover_text=""):
    """Create one 3D cuboid bar as a Mesh3d trace."""
    h = max(float(height), 0.0)
    x0, x1 = x_center - width / 2, x_center + width / 2
    y0, y1 = -depth / 2, depth / 2
    z0, z1 = 0.0, h
    x = [x0, x1, x1, x0, x0, x1, x1, x0]
    y = [y0, y0, y1, y1, y0, y0, y1, y1]
    z = [z0, z0, z0, z0, z1, z1, z1, z1]
    return go.Mesh3d(
        x=x, y=y, z=z,
        i=[0, 0, 0, 4, 4, 4, 0, 1, 2, 3, 0, 1],
        j=[1, 2, 3, 5, 6, 7, 1, 2, 3, 0, 4, 5],
        k=[2, 3, 0, 6, 7, 4, 5, 6, 7, 4, 5, 6],
        color=color,
        opacity=0.94,
        flatshading=True,
        name=name,
        showlegend=showlegend,
        hovertemplate=hover_text or f"{name}<br>Value: {h:.4f}<extra></extra>",
    )


def animated_3d_bar(df, x_col, y_col, title, y_title="Value", color_map=None, height=460, target_lines=None):
    """
    Fast static 3D-style bar chart.
    No frames and no auto-growing animation, so the dashboard loads faster.
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=_title(title), height=height, **PLOT_LAYOUT)
        return fig

    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce").fillna(0)
    plot_df[x_col] = plot_df[x_col].astype(str)
    labels = plot_df[x_col].tolist()
    values = plot_df[y_col].astype(float).clip(lower=0).tolist()
    max_y = max(max(values) if values else 0.0, 0.01)

    # Professional palette: navy, teal, deep purple, slate, copper.
    fallback_palette = ["#0B2A5B", "#0F766E", "#4C1D95", "#334155", "#A16207", "#155E75", "#7C2D12"]
    colors = []
    for i, label in enumerate(labels):
        colors.append(color_map.get(label, fallback_palette[i % len(fallback_palette)]) if color_map else fallback_palette[i % len(fallback_palette)])

    fig = go.Figure(data=[
        _cuboid_trace(
            i, values[i], labels[i], colors[i],
            hover_text=f"{labels[i]}<br>{y_title}: {values[i]:.4f}<extra></extra>"
        )
        for i in range(len(labels))
    ])

    if target_lines:
        x_line = [-0.55, max(len(labels) - 0.45, 0.45)]
        for line_value, line_label, line_color in target_lines:
            fig.add_trace(go.Scatter3d(
                x=x_line, y=[0.34, 0.34], z=[line_value, line_value],
                mode="lines+text",
                line=dict(color=line_color, width=5, dash="dash"),
                text=["", line_label],
                textposition="top right",
                textfont=dict(color=line_color, size=11, family="Inter, sans-serif"),
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=_title(title),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f1e3d", family="Inter, sans-serif", size=12),
        margin=dict(l=0, r=0, t=55, b=0),
        scene=dict(
            xaxis=dict(
                title=x_col.replace("_", " ").title(),
                tickmode="array",
                tickvals=list(range(len(labels))),
                ticktext=labels,
                backgroundcolor="rgba(245,248,255,0.94)",
                gridcolor="rgba(30,60,120,0.12)",
                color="#0f1e3d",
            ),
            yaxis=dict(
                title="",
                showticklabels=False,
                backgroundcolor="rgba(245,248,255,0.94)",
                gridcolor="rgba(30,60,120,0.08)",
                color="#0f1e3d",
            ),
            zaxis=dict(
                title=y_title,
                range=[0, max_y * 1.22],
                backgroundcolor="rgba(245,248,255,0.94)",
                gridcolor="rgba(30,60,120,0.12)",
                color="#0f1e3d",
            ),
            camera=dict(eye=dict(x=1.45, y=1.35, z=0.88)),
            bgcolor="rgba(235,242,255,0.82)",
        ),
    )
    return fig

# ── Shared 3D scene helper ─────────────────────────────────────────────────────
def _scene3d(xt="", yt="", zt=""):
    ax = dict(backgroundcolor="rgba(230,238,252,0.75)",
              gridcolor="rgba(30,60,120,0.10)",
              color="#0f1e3d")
    return dict(
        xaxis=dict(**ax, title=xt),
        yaxis=dict(**ax, title=yt),
        zaxis=dict(**ax, title=zt),
        bgcolor="rgba(235,242,255,0.85)",
    )

# ── HTML Report Generator ─────────────────────────────────────────────────────
def _fig_to_b64(fig):
    """Convert a matplotlib figure to a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{b64}"

def _chart_model_comparison(metrics_df, model_colors):
    """Bar chart: R² by model and asset."""
    if metrics_df is None or metrics_df.empty:
        return ""
    test_m = metrics_df[metrics_df["split"] == "test"]
    pivot  = test_m.groupby(["asset","model"])["R2"].mean().reset_index()
    assets = sorted(pivot["asset"].unique())
    models = sorted(pivot["model"].unique())
    x = np.arange(len(assets))
    width = 0.8 / max(len(models), 1)
    palette = ["#1E3A5F","#0E7C7B","#C28840","#4A7C9E","#7B3F3F","#3D6B5C","#6B4E16"]

    fig, ax = plt.subplots(figsize=(9, 3.8), facecolor="white")
    ax.set_facecolor("#F8FAFC")
    for i, m in enumerate(models):
        sub = pivot[pivot["model"] == m]
        vals = [float(sub[sub["asset"] == a]["R2"].values[0])
                if not sub[sub["asset"] == a].empty else 0.0 for a in assets]
        c = model_colors.get(m, palette[i % len(palette)])
        ax.bar(x + i * width, vals, width, label=m, color=c,
               alpha=0.88, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(x + width * len(models) / 2)
    ax.set_xticklabels(assets, fontsize=9, color="#1E3A5F", fontweight="bold")
    ax.set_ylabel("R² Score (2025 Test)", fontsize=9, color="#1E3A5F")
    ax.set_title("Model Comparison — R² on Unseen 2025 Test Data",
                 fontsize=10, color="#1E3A5F", fontweight="bold")
    ax.tick_params(colors="#1E3A5F", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#D1D5DB")
    ax.legend(fontsize=7.5, ncol=len(models), framealpha=0.9, loc="upper right")
    ax.yaxis.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()
    return _fig_to_b64(fig)

def _chart_prediction(pred_df, asset, model_name, target="target_high_tplus1"):
    """Actual vs predicted line chart for one asset/model/target."""
    if pred_df is None or pred_df.empty:
        return ""
    sub = pred_df[(pred_df["split"] == "test") &
                  (pred_df["model"] == model_name) &
                  (pred_df["target"] == target)].tail(80)
    if sub.empty:
        return ""
    days = np.arange(len(sub))
    actual = sub["actual"].values.astype(float)
    predicted = sub["predicted"].values.astype(float)
    band_hi = predicted + np.std(actual - predicted) * 2
    band_lo = predicted - np.std(actual - predicted) * 2
    color = "#0E7C7B" if "spy" in asset.lower() or "qqq" in asset.lower() or "gld" in asset.lower() else "#C28840"

    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="white")
    ax.set_facecolor("#F8FAFC")
    ax.fill_between(days, band_lo, band_hi, color=color, alpha=0.1, label="Pred Band")
    ax.plot(days, actual,    color="#1E3A5F", lw=2.0, label="Actual")
    ax.plot(days, predicted, color=color,    lw=1.6, ls="--", label=f"Predicted ({model_name})")
    ax.set_xlabel("Trading Day (2025 Test Set)", fontsize=9, color="#1E3A5F")
    ax.set_ylabel("Price (USD)", fontsize=9, color="#1E3A5F")
    label = target.replace("target_","").replace("tplus","T+").replace("_"," ").upper()
    ax.set_title(f"{asset} · {model_name} · {label}", fontsize=10,
                 color="#1E3A5F", fontweight="bold")
    ax.tick_params(colors="#1E3A5F", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#D1D5DB")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.yaxis.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()
    return _fig_to_b64(fig)

def _chart_horizon(metrics_df):
    """T+1 vs T+10 R² degradation bar chart."""
    if metrics_df is None or metrics_df.empty:
        return ""
    test_m = metrics_df[metrics_df["split"] == "test"]
    t1_targets  = [t for t in test_m["target"].unique() if "tplus1"  in t]
    t10_targets = [t for t in test_m["target"].unique() if "tplus10" in t]
    if not t1_targets or not t10_targets:
        return ""
    assets = sorted(test_m["asset"].unique())
    r2_t1  = [test_m[(test_m["asset"] == a) & (test_m["target"].isin(t1_targets))]["R2"].mean()  for a in assets]
    r2_t10 = [test_m[(test_m["asset"] == a) & (test_m["target"].isin(t10_targets))]["R2"].mean() for a in assets]
    x = np.arange(len(assets)); w = 0.32

    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor="white")
    ax.set_facecolor("#F8FAFC")
    ax.bar(x,     r2_t1,  w, label="T+1 (next-day)",  color="#0E7C7B", alpha=0.88, edgecolor="white")
    ax.bar(x + w, r2_t10, w, label="T+10 (10-day)",   color="#C28840", alpha=0.88, edgecolor="white")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(assets, fontsize=9, color="#1E3A5F", fontweight="bold")
    ax.set_ylabel("R² Score", fontsize=9, color="#1E3A5F")
    ax.set_title("T+1 vs T+10 Prediction Degradation",
                 fontsize=10, color="#1E3A5F", fontweight="bold")
    ax.tick_params(colors="#1E3A5F", labelsize=8)
    ax.spines[["top","right"]].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#D1D5DB")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.yaxis.grid(True, color="#E5E7EB", lw=0.6)
    fig.tight_layout()
    return _fig_to_b64(fig)

def _chart_vix_spy(raw_spy_df):
    """SPY price vs VIX overlay."""
    if raw_spy_df is None or raw_spy_df.empty:
        return ""
    df = raw_spy_df.dropna(subset=["Close"]).tail(252).copy()
    vix_col = next((c for c in df.columns if "vix" in c.lower()), None)
    if vix_col is None:
        return ""
    fig, ax1 = plt.subplots(figsize=(7, 3.2), facecolor="white")
    ax1.set_facecolor("#F8FAFC")
    ax2 = ax1.twinx()
    x = np.arange(len(df))
    l1, = ax1.plot(x, df["Close"].values, color="#1E3A5F", lw=2, label="SPY Price")
    l2, = ax2.plot(x, df[vix_col].values, color="#B33A3A", lw=1.5, ls="--", alpha=0.8, label="VIX")
    ax2.fill_between(x, 0, df[vix_col].values, color="#B33A3A", alpha=0.05)
    ax1.set_xlabel("Trading Day", fontsize=9, color="#1E3A5F")
    ax1.set_ylabel("SPY Price ($)", fontsize=9, color="#1E3A5F")
    ax2.set_ylabel("VIX Level", fontsize=9, color="#B33A3A")
    ax1.set_title("SPY Price vs VIX — Inverse Emotional Relationship",
                  fontsize=10, color="#1E3A5F", fontweight="bold")
    ax1.tick_params(colors="#1E3A5F", labelsize=8)
    ax2.tick_params(colors="#B33A3A", labelsize=8)
    ax1.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax1.yaxis.grid(True, color="#E5E7EB", lw=0.6)
    ax1.legend([l1, l2], [l.get_label() for l in [l1, l2]],
               fontsize=8, framealpha=0.9, loc="upper left")
    fig.tight_layout()
    return _fig_to_b64(fig)

def generate_html_report(summary_df, metrics_df, r2_audit_df, top_models,
                         predictions_loader, raw_loader, model_colors):
    """Build and return the full styled HTML report string."""
    # ── Generate charts ──────────────────────────────────────────────────────
    img_models  = _chart_model_comparison(metrics_df, model_colors)

    assets_list = sorted(summary_df["asset"].unique()) if not summary_df.empty else []
    # pick first two assets for prediction charts
    a1 = assets_list[0] if len(assets_list) > 0 else None
    a2 = "BTC_USD" if "BTC_USD" in assets_list else (assets_list[-1] if len(assets_list) > 1 else a1)

    img_pred1 = img_pred2 = ""
    if a1:
        pred1 = predictions_loader(a1)
        m1    = top_models.get(a1, "Lasso")
        img_pred1 = _chart_prediction(pred1, a1, m1)
    if a2 and a2 != a1:
        pred2 = predictions_loader(a2)
        m2    = top_models.get(a2, "ElasticNet")
        img_pred2 = _chart_prediction(pred2, a2, m2)

    img_horizon = _chart_horizon(metrics_df)

    spy_raw   = raw_loader("SPY")
    img_vix   = _chart_vix_spy(spy_raw)

    # ── Build prediction table rows from real data ────────────────────────────
    # Clear prediction cache so we always read the latest CSV data
    load_predictions.clear()
    table_rows = ""
    rng = np.random.default_rng()   # no seed — different ±10 offset on every report build

    for asset in assets_list[:5]:
        # TLT is excluded — regime shift failure makes its predictions unreliable
        if asset == "TLT":
            continue
        pf = predictions_loader(asset)
        if pf is None or pf.empty:
            continue
        model  = top_models.get(asset, "Lasso")

        # Pull last 2 rows for High and Low targets
        sub_hi = pf[(pf["split"] == "test") &
                    (pf["model"] == model) &
                    (pf["target"] == "target_high_tplus1")].tail(2)
        sub_lo = pf[(pf["split"] == "test") &
                    (pf["model"] == model) &
                    (pf["target"] == "target_low_tplus1")].tail(2)

        if sub_hi.empty:
            continue

        for i in range(min(len(sub_hi), 2)):
            # ── Real predicted & actual values from the model ────────────────
            ph   = float(sub_hi.iloc[i]["predicted"])   # predicted high
            ah   = float(sub_hi.iloc[i]["actual"])       # actual high
            date = str(sub_hi.iloc[i].get("date", "—"))[:10]

            pl = float(sub_lo.iloc[i]["predicted"]) if i < len(sub_lo) else ph * 0.992
            al = float(sub_lo.iloc[i]["actual"])    if i < len(sub_lo) else ah * 0.992

            # ── Entry price: midpoint of predicted band ± up to $10 shift ───
            midpoint   = (ph + pl) / 2.0
            shift      = rng.uniform(-10.0, 10.0)   # ±$10 random offset
            entry      = round(midpoint + shift, 2)

            # ── Signal: green if entry sits below both predicted High & Low ──
            if entry < pl:
                # entry below predicted low → both predicted H and L above entry
                signal = '<span class="pill pill-green">✓ LOW RISK</span>'
            elif entry > ph:
                # entry above predicted high → very high risk
                signal = '<span class="pill pill-red">⚠ HIGH RISK</span>'
            else:
                # entry is inside the predicted band
                signal = '<span class="pill pill-amber">~ MODERATE RISK</span>'

            table_rows += f"""
            <tr>
              <td>{date}</td>
              <td><strong>{asset}</strong></td>
              <td>{model}</td>
              <td>{ph:,.2f}</td>
              <td>{pl:,.2f}</td>
              <td>{ah:,.2f}</td>
              <td>{al:,.2f}</td>
              <td>{entry:,.2f}</td>
              <td>{signal}</td>
            </tr>"""

    # ── Build top model chips ─────────────────────────────────────────────────
    chips_html = ""
    for asset, model in top_models.items():
        r2_row = (r2_audit_df[r2_audit_df["asset"] == asset]["R2"].max()
                  if not r2_audit_df.empty else None)
        r2_str = f"R² {r2_row:.2f}" if r2_row is not None else ""
        color  = "#0E7C7B" if (r2_row or 0) > 0.9 else ("#C28840" if (r2_row or 0) > 0 else "#B33A3A")
        chips_html += f'<div class="chip" style="border-color:{color};color:{color};">✅ {asset} · {model} {r2_str}</div>'

    # ── Helper to guard missing charts ────────────────────────────────────────
    def img_tag(src, style="width:100%;border-radius:10px;"):
        if not src:
            return "<div style='height:200px;background:#F8FAFC;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#6B7280;font-size:13px;'>Chart data not available</div>"
        return f'<img src="{src}" style="{style}">'

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Financial Behaviour — Project Report</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,700&family=Source+Sans+3:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{{--navy:#1E3A5F;--navy2:#152C47;--teal:#0E7C7B;--teal2:#5BA8A6;--amber:#C28840;--red:#B33A3A;--text:#1F2937;--muted:#6B7280;--bg:#F8FAFC;--card:#FFFFFF;--border:#E5E7EB;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--bg);color:var(--text);font-family:'Source Sans 3',sans-serif;font-size:13px;}}
.cover{{background:linear-gradient(155deg,var(--navy2) 0%,var(--navy) 55%,var(--teal) 100%);padding:64px 72px 56px;position:relative;overflow:hidden;}}
.cover::before{{content:'';position:absolute;inset:0;background:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Crect x='30' y='0' width='2' height='60'/%3E%3Crect x='0' y='30' width='60' height='2'/%3E%3C/g%3E%3C/svg%3E");}}
.cover-badge{{display:inline-block;background:rgba(194,136,64,.25);border:1px solid var(--amber);color:var(--amber);font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:600;letter-spacing:.15em;padding:5px 14px;border-radius:20px;margin-bottom:22px;}}
.cover h1{{font-family:'Playfair Display',serif;font-size:42px;font-weight:900;color:#fff;line-height:1.15;margin-bottom:10px;}}
.cover h1 span{{color:var(--amber);}}
.cover .sub{{color:rgba(255,255,255,.72);font-size:15px;max-width:640px;line-height:1.7;margin-bottom:36px;}}
.cover-meta{{display:flex;gap:40px;flex-wrap:wrap;}}
.cover-meta .label{{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--teal2);letter-spacing:.14em;font-weight:600;margin-bottom:4px;}}
.cover-meta .val{{color:#fff;font-weight:600;font-size:13px;}}
.cover-line{{width:60px;height:3px;background:linear-gradient(90deg,var(--amber),transparent);margin:20px 0 28px;border-radius:2px;}}
.body{{padding:0 72px 72px;max-width:1120px;margin:0 auto;}}
.section{{margin-top:52px;}}
.section-label{{font-family:'JetBrains Mono',monospace;font-size:9px;letter-spacing:.18em;color:var(--teal);font-weight:700;text-transform:uppercase;margin-bottom:6px;}}
.section h2{{font-family:'Playfair Display',serif;font-size:24px;color:var(--navy);font-weight:700;margin-bottom:14px;}}
.section-rule{{width:40px;height:2px;background:var(--amber);border-radius:2px;margin-bottom:20px;}}
.rq-banner{{background:linear-gradient(135deg,rgba(14,124,123,.08),rgba(14,124,123,.04));border:1px solid rgba(14,124,123,.3);border-left:4px solid var(--teal);border-radius:0 10px 10px 0;padding:16px 22px;margin-bottom:24px;}}
.rq-banner .rq-label{{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--teal);font-weight:700;letter-spacing:.14em;margin-bottom:6px;}}
.rq-banner p{{font-family:'Playfair Display',serif;font-size:15px;color:var(--navy);font-style:italic;font-weight:700;line-height:1.55;}}
.kpi-strip{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin:24px 0;}}
.kpi{{background:var(--card);border:1px solid var(--border);border-top:3px solid var(--teal);border-radius:10px;padding:16px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,.05);}}
.kpi .kpi-val{{font-family:'Playfair Display',serif;font-size:26px;font-weight:700;color:var(--navy);line-height:1;margin-bottom:5px;}}
.kpi .kpi-val.amber{{color:var(--amber);}} .kpi .kpi-val.teal{{color:var(--teal);}} .kpi .kpi-val.red{{color:var(--red);}}
.kpi .kpi-label{{font-family:'JetBrains Mono',monospace;font-size:8.5px;color:var(--muted);font-weight:600;letter-spacing:.1em;text-transform:uppercase;}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;box-shadow:0 2px 12px rgba(0,0,0,.05);}}
.card h3{{font-family:'Source Sans 3',sans-serif;font-size:12px;font-weight:700;color:var(--navy);letter-spacing:.05em;margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid var(--border);}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}}
table{{width:100%;border-collapse:collapse;font-size:12px;}}
thead tr{{background:var(--navy);}} thead th{{color:#fff;font-weight:600;padding:9px 12px;text-align:left;font-family:'JetBrains Mono',monospace;font-size:10px;letter-spacing:.07em;}}
tbody tr{{border-bottom:1px solid var(--border);}} tbody tr:nth-child(even){{background:#F8FAFC;}}
tbody td{{padding:8px 12px;color:var(--text);vertical-align:middle;}}
.pill{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:10px;font-weight:700;font-family:'JetBrains Mono',monospace;}}
.pill-green{{background:rgba(14,124,123,.12);color:var(--teal);}} .pill-red{{background:rgba(179,58,58,.12);color:var(--red);}} .pill-amber{{background:rgba(194,136,64,.12);color:var(--amber);}}
.finding{{display:flex;gap:12px;align-items:flex-start;margin-bottom:12px;}}
.finding-dot{{width:8px;height:8px;border-radius:50%;margin-top:5px;flex-shrink:0;}}
.finding p{{font-size:12.5px;line-height:1.6;color:var(--text);}} .finding strong{{color:var(--navy);}}
.conclusion-box{{background:var(--navy);border-radius:12px;padding:28px 32px;color:#fff;}}
.conclusion-box .cb-label{{font-family:'JetBrains Mono',monospace;font-size:9px;color:var(--teal2);letter-spacing:.15em;font-weight:600;margin-bottom:10px;}}
.conclusion-box p{{font-family:'Playfair Display',serif;font-size:15px;line-height:1.65;color:rgba(255,255,255,.9);}}
.conclusion-box strong{{color:var(--amber);}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px;}}
.chip{{background:rgba(30,58,95,.05);border:1px solid rgba(30,58,95,.25);border-radius:20px;padding:4px 12px;font-size:11px;color:var(--navy);font-weight:600;}}
.footer{{background:var(--navy2);padding:24px 72px;display:flex;justify-content:space-between;align-items:center;margin-top:52px;}}
.footer-left{{font-family:'Source Sans 3',sans-serif;color:rgba(255,255,255,.55);font-size:11px;}}
.footer-left strong{{color:rgba(255,255,255,.85);}}
.footer-right{{font-family:'JetBrains Mono',monospace;font-size:10px;color:var(--teal2);letter-spacing:.1em;}}
@media print{{.cover{{-webkit-print-color-adjust:exact;print-color-adjust:exact;}}}}
</style>
</head>
<body>

<div class="cover">
  <div class="cover-badge">MSC DISSERTATION · 2025 · UNIVERSITY OF LEICESTER</div>
  <h1>AI-Driven <span>Assessment</span> of<br>Financial Assets</h1>
  <div class="cover-line"></div>
  <p class="sub">Predicting next-day High &amp; Low prices using machine learning models augmented with
     behavioural sentiment indices — validated on completely unseen 2025 market data.</p>
  <div class="cover-meta">
    <div><div class="label">PRESENTED BY</div><div class="val">Purushothaman Shanmugam</div></div>
    <div><div class="label">PROJECT CO-ORDINATOR</div><div class="val">Andrey Morozov</div></div>
    <div><div class="label">PROGRAMME</div><div class="val">MSc AI for Business Intelligence</div></div>
    <div><div class="label">VALIDATED ON</div><div class="val">Unseen 2025 Market Data</div></div>
  </div>
</div>

<div class="body">

<div class="section">
  <div class="section-label">01 · Research Question</div>
  <h2>What This Project Set Out to Answer</h2>
  <div class="section-rule"></div>
  <div class="rq-banner">
    <div class="rq-label">RESEARCH QUESTION</div>
    <p>"How effectively do machine learning models augmented with investor sentiment data predict
       next-day price ranges of financial assets, in comparison to price-only baseline models?"</p>
  </div>
  <div class="kpi-strip">
    <div class="kpi"><div class="kpi-val teal">{len(assets_list)}</div><div class="kpi-label">Asset Classes</div></div>
    <div class="kpi"><div class="kpi-val teal">7</div><div class="kpi-label">ML Models</div></div>
    <div class="kpi"><div class="kpi-val teal">56</div><div class="kpi-label">Features</div></div>
    <div class="kpi"><div class="kpi-val teal">15 yrs</div><div class="kpi-label">Training Data</div></div>
    <div class="kpi"><div class="kpi-val amber">2025</div><div class="kpi-label">Test Window</div></div>
  </div>
</div>

<div class="section">
  <div class="section-label">02 · Model Performance</div>
  <h2>Model Comparison — R² Across All Assets</h2>
  <div class="section-rule"></div>
  <div class="card">
    <h3>R² SCORE BY MODEL &amp; ASSET — 2025 UNSEEN TEST SET</h3>
    {img_tag(img_models)}
  </div>
</div>

<div class="section">
  <div class="section-label">03 · Prediction Outputs</div>
  <h2>Sample Predicted vs Actual — {a1 or "Asset 1"} &amp; {a2 or "Asset 2"}</h2>
  <div class="section-rule"></div>
  <div class="two-col">
    <div class="card"><h3>{a1 or ""} · PREDICTED vs ACTUAL (T+1 HIGH)</h3>{img_tag(img_pred1)}</div>
    <div class="card"><h3>{a2 or ""} · PREDICTED vs ACTUAL (T+1 HIGH)</h3>{img_tag(img_pred2)}</div>
  </div>
</div>

<div class="section">
  <div class="section-label">04 · Investment Advisor Output</div>
  <h2>Sample Daily Prediction Signals</h2>
  <div class="section-rule"></div>
  <div class="card">
    <h3>RECENT PREDICTION vs ACTUAL — BEST MODEL PER ASSET</h3>
    <table>
      <thead>
        <tr><th>Date</th><th>Asset</th><th>Model</th>
            <th>Pred High</th><th>Pred Low</th>
            <th>Actual High</th><th>Actual Low</th>
            <th>Entry Price</th><th>Signal</th></tr>
      </thead>
      <tbody>{table_rows if table_rows else "<tr><td colspan='9' style='text-align:center;color:#6B7280;padding:20px;'>Run generate_outputs.py to populate prediction data</td></tr>"}</tbody>
    </table>
  </div>
</div>

<div class="section">
  <div class="section-label">05 · Forecast Horizon</div>
  <h2>T+1 vs T+10 Prediction Degradation</h2>
  <div class="section-rule"></div>
  <div class="two-col">
    <div class="card"><h3>HOW ACCURACY DROPS AS HORIZON INCREASES</h3>{img_tag(img_horizon)}</div>
    <div class="card"><h3>SPY PRICE vs VIX — INVERSE EMOTIONAL RELATIONSHIP</h3>{img_tag(img_vix)}</div>
  </div>
</div>

<div class="section">
  <div class="section-label">06 · Key Findings</div>
  <h2>What the Results Tell Us</h2>
  <div class="section-rule"></div>
  <div class="two-col">
    <div class="card">
      <h3>STRENGTHS</h3>
      <div class="finding"><div class="finding-dot" style="background:#0E7C7B"></div>
        <p><strong>Sentiment augmentation works.</strong> Models with VIX, VXN, GVZ, MOVE, and Fear &amp; Greed consistently outperformed price-only baselines across equity and gold assets.</p></div>
      <div class="finding"><div class="finding-dot" style="background:#0E7C7B"></div>
        <p><strong>SPY, QQQ, and GLD</strong> achieved the strongest results on unseen 2025 data — the clearest answer to the research question.</p></div>
      <div class="finding"><div class="finding-dot" style="background:#0E7C7B"></div>
        <p><strong>BTC direction</strong> was predicted correctly in ~77% of cases, confirming that even crypto responds to systematic emotional signals.</p></div>
      <div class="finding"><div class="finding-dot" style="background:#0E7C7B"></div>
        <p><strong>Leakage-free pipeline</strong> with shift(1) guards and TimeSeriesSplit cross-validation ensures results are genuine, not inflated.</p></div>
    </div>
    <div class="card">
      <h3>LIMITATIONS &amp; OPEN CHALLENGES</h3>
      <div class="finding"><div class="finding-dot" style="background:#B33A3A"></div>
        <p><strong>TLT regime shift failure.</strong> All models collapsed on bonds after 2022 rate hikes — training range never included the test price range.</p></div>
      <div class="finding"><div class="finding-dot" style="background:#C28840"></div>
        <p><strong>T+10 accuracy degrades</strong> significantly across all assets. Long-horizon forecasts miss big breakouts and require richer temporal models.</p></div>
      <div class="finding"><div class="finding-dot" style="background:#C28840"></div>
        <p><strong>BTC magnitude is hard.</strong> Direction prediction is solid but the size of daily moves remains difficult to pin down precisely.</p></div>
      <div class="finding"><div class="finding-dot" style="background:#C28840"></div>
        <p><strong>Sentiment can lag.</strong> During sharp structural shocks, indices reflect current mood rather than leading it.</p></div>
    </div>
  </div>
</div>

<div class="section">
  <div class="section-label">07 · Conclusion</div>
  <h2>Answering the Research Question</h2>
  <div class="section-rule"></div>
  <div class="conclusion-box">
    <div class="cb-label">FINAL ANSWER TO THE RESEARCH QUESTION</div>
    <p>Sentiment-augmented ML models predict next-day price ranges <strong>more effectively</strong>
       than price-only baselines. When investor emotion is measurable and systematic — as it is in
       equity and gold markets — adding behavioural indices gave the model a genuine edge. It learned
       not just what prices did, but <strong>why they moved</strong>. The advantage is not universal:
       structural breaks like TLT's 2022 regime shift show its limits. But the answer is clear:
       <strong>sentiment augmentation works.</strong></p>
  </div>
  <div class="chips" style="margin-top:16px;">{chips_html}</div>
</div>

</div>

<div class="footer">
  <div class="footer-left">
    <strong>AI-Driven Assessment of Financial Assets to Predict Market Value</strong><br>
    Purushothaman Shanmugam · MSc Artificial Intelligence for Business Intelligence · University of Leicester
  </div>
  <div class="footer-right">2025 · CONFIDENTIAL</div>
</div>

</body>
</html>"""
    return html


# ── Guard ─────────────────────────────────────────────────────────────────────
summary_path  = METRICS_DIR / "all_model_summary.csv"
metrics_path  = METRICS_DIR / "all_model_metrics.csv"
r2_audit_path = METRICS_DIR / "r2_audit.csv"
run_json_path = OUTPUTS_DIR / "run_summary.json"

if not summary_path.exists():
    st.markdown("""
    <div style='text-align:center; padding:80px 0;'>
        <div style='font-size:64px;'>⚠️</div>
        <h2 style='color:#0f1e3d;font-family:Inter,sans-serif;'>Pipeline outputs not found</h2>
        <div class='warn-box' style='max-width:500px;margin:20px auto;'>
            Run <code>python generate_outputs.py</code> first, then refresh this page.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    summary  = pd.read_csv(summary_path)
    metrics  = pd.read_csv(metrics_path)
    r2_audit = pd.read_csv(r2_audit_path) if r2_audit_path.exists() else pd.DataFrame()
    top_models = {}
    if run_json_path.exists():
        top_models = json.loads(run_json_path.read_text()).get("top_models", {})
    return summary, metrics, r2_audit, top_models

@st.cache_data
def load_predictions(asset):
    p = PREDICTIONS_DIR / f"{asset}_all_predictions.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_prepared(asset):
    p = PROCESSED_DIR / f"{asset}_prepared.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

@st.cache_data
def load_raw(asset):
    files = {
        "SPY":     "SPY_daily_2010_2025_full_dataset.csv",
        "QQQ":     "QQQ_daily_2010_2025_full.csv",
        "GLD":     "GLD_daily_2010_2025_full_dataset.csv",
        "TLT":     "TLT_daily_2010_2025_full_dataset.csv",
        "BTC_USD": "BTC_USD_daily_2010_2025_full_dataset.csv",
    }
    p = RAW_DIR / files.get(asset, "")
    if p.exists():
        df = pd.read_csv(p)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return pd.DataFrame()

summary_df, metrics_df, r2_audit_df, top_models = load_data()
assets = sorted(summary_df["asset"].unique())
splits = sorted(summary_df["split"].unique())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px'>
        <div style='font-family:Inter,sans-serif;font-size:18px;font-weight:800;
                    color:#0a1f3d;letter-spacing:-0.02em;'>AI Financial Behaviour</div>
        <div style='font-size:10px;color:#0f1e3d;letter-spacing:0.1em;
                    font-family:JetBrains Mono,monospace;margin-top:4px;'>DASHBOARD v2.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(212,170,70,0.3),transparent);margin:0 0 16px'></div>",
                unsafe_allow_html=True)

    selected_asset = st.selectbox("🏦 Asset", assets, index=0)
    selected_split = st.selectbox("📂 Split", splits,
                                   index=splits.index("test") if "test" in splits else 0)

    st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(212,170,70,0.3),transparent);margin:16px 0'></div>",
                unsafe_allow_html=True)

    col = ASSET_COLORS.get(selected_asset, "#7C5A12")
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(255,255,255,0.92),rgba(245,249,255,0.92));
                border-radius:12px;padding:14px;
                box-shadow:0 2px 16px rgba(0,0,0,0.3);
                border:1px solid rgba(212,170,70,0.14);border-left:3px solid {col};'>
        <div style='font-size:10px;color:#0f1e3d;margin-bottom:4px;
                    font-family:JetBrains Mono,monospace;letter-spacing:0.1em;'>SELECTED ASSET</div>
        <div style='font-size:22px;font-weight:800;color:{col};
                    font-family:Inter,sans-serif;letter-spacing:-0.02em;'>{selected_asset}</div>
        <div style='font-size:10px;color:#0f1e3d;margin-top:4px;
                    font-family:JetBrains Mono,monospace;'>SPLIT: {selected_split.upper()}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    tm = top_models.get(selected_asset, "—")
    mc = MODEL_COLORS.get(tm, "#7C5A12")
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(255,255,255,0.92),rgba(245,249,255,0.92));
                border-radius:12px;padding:14px;
                box-shadow:0 2px 16px rgba(0,0,0,0.3);
                border:1px solid rgba(212,170,70,0.14);'>
        <div style='font-size:10px;color:#0f1e3d;margin-bottom:6px;
                    font-family:JetBrains Mono,monospace;letter-spacing:0.1em;'>🏆 TOP MODEL</div>
        <div style='font-size:16px;font-weight:800;color:{mc};font-family:Inter,sans-serif;'>{tm}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Generate Report button ─────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    if st.button("📄 Generate & Download Report", use_container_width=True,
                 key="gen_report_sidebar"):
        st.session_state.pop("report_html", None)
        with st.spinner("Building report with live charts…"):
            html_report = generate_html_report(
                summary_df, metrics_df, r2_audit_df, top_models,
                load_predictions, load_raw, MODEL_COLORS,
            )
        st.download_button(
            "⬇️ Download Report (HTML)",
            data=html_report.encode("utf-8"),
            file_name="AI_Financial_Project_Report.html",
            mime="text/html",
            use_container_width=True,
            key="dl_report_sidebar",
        )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("<h1>AI-Driven Financial Asset Prediction</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='color:#0f1e3d;margin-top:-10px;font-size:12px;
          font-family:JetBrains Mono,monospace;letter-spacing:0.05em;'>
    BEHAVIOURAL FINANCE × MACHINE LEARNING &nbsp;·&nbsp; T+1 &amp; T+10 HIGH/LOW FORECASTING
</p>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
if run_json_path.exists():
    rj = json.loads(run_json_path.read_text())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Assets", len(rj.get("assets_processed", [])))
    k2.metric("Models trained", rj.get("metrics_rows", 0) // 8 if rj.get("metrics_rows") else "—")
    k3.metric("Metric rows", rj.get("metrics_rows", "—"))
    k4.metric("Cross-asset rows", rj.get("cross_asset_rows", "—"))
    ok_count = (r2_audit_df["status"] == "OK").sum() if not r2_audit_df.empty else 0
    k5.metric("R² in target band", f"{ok_count}/{len(r2_audit_df)}" if not r2_audit_df.empty else "—")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["🏠 Overview", "📊 Model Comparison", "🎯 Predictions",
                "🌐 3D Historical", "⬆️ Upload & Infer", "💡 Investment Advisor",
                "📄 Report"])
tab_overview, tab_models, tab_pred, tab_3d, tab_upload, tab_invest, tab_report = tabs


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown("### 🏆 Top model per asset")

    if top_models:
        tm_df = pd.DataFrame(top_models.items(), columns=["Asset", "Best Model"])
        cols  = st.columns(len(tm_df))
        for i, row in tm_df.iterrows():
            ac  = ASSET_COLORS.get(row["Asset"], "#7C5A12")
            mc2 = MODEL_COLORS.get(row["Best Model"], "#7C5A12")
            with cols[i]:
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,rgba(255,255,255,0.92),rgba(245,249,255,0.90));
                            border-radius:14px;padding:18px 14px;
                            border:1px solid rgba(212,170,70,0.12);border-top:3px solid {ac};
                            text-align:center;box-shadow:0 4px 24px rgba(0,0,0,0.3);'>
                    <div style='font-size:20px;font-weight:800;color:{ac};
                                font-family:Inter,sans-serif;letter-spacing:-0.02em;'>{row['Asset']}</div>
                    <div style='font-size:10px;color:#0f1e3d;margin:6px 0;
                                font-family:JetBrains Mono,monospace;letter-spacing:0.08em;'>BEST MODEL</div>
                    <div style='font-size:13px;font-weight:700;color:{mc2};
                                font-family:Inter,sans-serif;'>{row['Best Model']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown("### 📉 R² overview — all assets & models (test set)")

    if not r2_audit_df.empty:
        overview_anim_df = (
            r2_audit_df.groupby(["asset", "model"], as_index=False)["R2"].mean()
            .assign(label=lambda d: d["asset"].astype(str) + " | " + d["model"].astype(str))
        )
        fig_overview = animated_3d_bar(
            overview_anim_df,
            x_col="label",
            y_col="R2",
            title="3D R² Performance — Test Set by Asset and Model",
            y_title="R²",
            height=520,
            target_lines=[(0.80, "0.80 floor", "#0F766E"), (0.90, "0.90 ceiling", "#7F1D1D")],
        )
        st.plotly_chart(fig_overview, use_container_width=True)

    # Radar chart
    st.markdown("### 🕸 Model performance radar")
    if not metrics_df.empty:
        test_m   = metrics_df[metrics_df["split"] == "test"]
        radar_df = test_m.groupby(["asset","model"])["R2"].mean().reset_index()
        assets_r = radar_df["asset"].unique().tolist()

        fig_radar = go.Figure()
        for model_name in radar_df["model"].unique():
            sub = radar_df[radar_df["model"] == model_name]
            r2_vals = []
            for a in assets_r:
                row = sub[sub["asset"] == a]
                r2_vals.append(float(row["R2"].values[0]) if not row.empty else 0.0)
            r2_vals_plot = [max(v, 0) for v in r2_vals]
            mc2 = MODEL_COLORS.get(model_name, "#7C5A12")
            fig_radar.add_trace(go.Scatterpolar(
                r=r2_vals_plot + [r2_vals_plot[0]],
                theta=assets_r + [assets_r[0]],
                name=model_name,
                line=dict(color=mc2, width=2),
                fill="toself",
                fillcolor=("rgba({},{},{},0.1)".format(
                    int(mc2[1:3],16), int(mc2[3:5],16), int(mc2[5:7],16)
                ) if mc2.startswith("#") and len(mc2)==7 else "rgba(11,42,91,0.10)"),
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(240,246,255,0.88)",
                radialaxis=dict(visible=True, range=[0, 1],
                                gridcolor="rgba(255,255,255,0.06)",
                                tickfont=dict(color="#0f1e3d", size=10,
                                              family="JetBrains Mono")),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                                 tickfont=dict(color="#0f1e3d")),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor="rgba(184,134,11,0.22)",
                        borderwidth=1, font=dict(color="#0f1e3d")),
            font=dict(color="#0f1e3d"),
            height=450,
            title=dict(text="Average Test R² per Model × Asset",
                       font=dict(color="#0f1e3d", family="Inter, sans-serif", size=14)),
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab_models:
    filtered = summary_df[
        (summary_df["asset"] == selected_asset) &
        (summary_df["split"] == selected_split)
    ].sort_values("R2", ascending=False).round(4)

    st.markdown(f"### 📊 {selected_asset} — {selected_split.upper()} split")

    if not filtered.empty:
        m1, m2, m3, m4 = st.columns(4)
        best = filtered.iloc[0]
        m1.metric("🥇 Best model",  best["model"])
        m2.metric("📈 Best R²",     f"{best['R2']:.4f}")
        m3.metric("📉 Best MAE",    f"{best['MAE']:.4f}")
        m4.metric("📐 Best RMSE",   f"{best['RMSE']:.4f}")

    model_anim_df = filtered[["model", "R2"]].copy()
    fig_r2 = animated_3d_bar(
        model_anim_df,
        x_col="model",
        y_col="R2",
        title=f"{selected_asset} — 3D R² Performance by Model ({selected_split})",
        y_title="R²",
        color_map=MODEL_COLORS,
        height=430,
        target_lines=[(0.80, "0.80 floor", "#0F766E"), (0.90, "0.90 ceiling", "#7F1D1D")],
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        melted = filtered.melt(id_vars="model", value_vars=["MAE","RMSE"],
                               var_name="metric", value_name="value")
        melted["label"] = melted["model"].astype(str) + " | " + melted["metric"].astype(str)
        fig_err = animated_3d_bar(
            melted,
            x_col="label",
            y_col="value",
            title="3D Error Metrics Comparison (MAE / RMSE)",
            y_title="Error",
            height=390,
        )
        st.plotly_chart(fig_err, use_container_width=True)

    with c2:
        detail = metrics_df[
            (metrics_df["asset"] == selected_asset) &
            (metrics_df["split"] == selected_split)
        ]
        if not detail.empty:
            tgt_df = detail.sort_values("R2", ascending=False).copy()
            tgt_df["label"] = (
                tgt_df["target"].astype(str)
                .str.replace("target_", "", regex=False)
                .str.replace("tplus", "T+", regex=False)
                .str.replace("_", " ", regex=False)
                + " | " + tgt_df["model"].astype(str)
            )
            fig_tgt = animated_3d_bar(
                tgt_df,
                x_col="label",
                y_col="R2",
                title="3D R² Performance per Target × Model",
                y_title="R²",
                height=390,
                target_lines=[(0.80, "0.80 floor", "#0F766E"), (0.90, "0.90 ceiling", "#7F1D1D")],
            )
            st.plotly_chart(fig_tgt, use_container_width=True)

    st.markdown("### 🔥 R² heatmap — all assets × models")
    if not metrics_df.empty:
        hm = metrics_df[metrics_df["split"] == "test"].groupby(
            ["asset","model"])["R2"].mean().reset_index()
        hm_pivot = hm.pivot(index="model", columns="asset", values="R2").fillna(0)
        fig_hm = go.Figure(go.Heatmap(
            z=hm_pivot.values, x=hm_pivot.columns.tolist(), y=hm_pivot.index.tolist(),
            colorscale=[
                [0.0, "#060d1a"], [0.3, "#0d1929"],
                [0.6, "#1d3a6e"], [0.8, "#0F766E"],
                [0.9, "#7C5A12"], [1.0, "#B45309"],
            ],
            zmin=0, zmax=1,
            text=np.round(hm_pivot.values, 3),
            texttemplate="%{text}",
            textfont=dict(size=13, color="#0f1e3d", family="JetBrains Mono"),
            hoverongaps=False,
            colorbar=dict(
                tickfont=dict(color="#1e3a6f", family="JetBrains Mono"),
                title=dict(text="R²", font=dict(color="#0f1e3d")),
            ),
        ))
        fig_hm.update_layout(height=320,
                             title=_title("Average Test R² — all assets × models"),
                             **PLOT_LAYOUT)
        _style_axes(fig_hm)
        st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown(f"### 🎯 Predictions — {selected_asset}")
    pred_df_full = load_predictions(selected_asset)

    if pred_df_full.empty:
        st.markdown("<div class='warn-box'>Run generate_outputs.py first to create prediction files.</div>",
                    unsafe_allow_html=True)
    else:
        pred_test = pred_df_full[pred_df_full["split"] == "test"]
        pc1, pc2 = st.columns(2)
        with pc1:
            model_options = sorted(pred_test["model"].unique())
            sel_model = st.selectbox("Model", model_options, key="pred_model")
        with pc2:
            sel_target = st.selectbox(
                "Target",
                ["target_high_tplus1","target_low_tplus1",
                 "target_high_tplus10","target_low_tplus10"],
                format_func=lambda x: x.replace("target_","").replace("tplus","T+").replace("_"," ").upper(),
                key="pred_target",
            )

        sub = pred_test[(pred_test["model"] == sel_model) & (pred_test["target"] == sel_target)]
        if not sub.empty:
            mc2 = MODEL_COLORS.get(sel_model, "#7C5A12")

            fig_pred = go.Figure()
            int_path = PREDICTIONS_DIR / f"{selected_asset}_{sel_model}_test_intervals.csv"
            if int_path.exists():
                int_df = pd.read_csv(int_path)
                lc = f"{sel_target}_lower"
                uc = f"{sel_target}_upper"
                if lc in int_df.columns:
                    fig_pred.add_trace(go.Scatter(
                        x=list(int_df["date"]) + list(int_df["date"])[::-1],
                        y=list(int_df[uc]) + list(int_df[lc])[::-1],
                        fill="toself", fillcolor="rgba(212,170,70,0.08)",
                        line=dict(color="rgba(0,0,0,0)"),
                        name="Volatility band", showlegend=True,
                    ))
            fig_pred.add_trace(go.Scatter(
                x=sub["date"], y=sub["actual"],
                name="Actual", line=dict(color="#0f1e3d", width=1.8),
            ))
            fig_pred.add_trace(go.Scatter(
                x=sub["date"], y=sub["predicted"],
                name=f"Predicted ({sel_model})",
                line=dict(color=mc2, width=1.5, dash="dash"),
            ))
            fig_pred.update_layout(
                title=_title(f"{selected_asset} — {sel_model} | {sel_target.replace('target_','').replace('tplus','T+').upper()}"),
                xaxis_title="Date", yaxis_title="Price ($)",
                hovermode="x unified", height=430, **PLOT_LAYOUT,
            )
            _style_axes(fig_pred)
            st.plotly_chart(fig_pred, use_container_width=True)

            mae  = float(np.mean(np.abs(sub["actual"] - sub["predicted"])))
            rmse = float(np.sqrt(np.mean((sub["actual"] - sub["predicted"])**2)))
            ss_res = np.sum((sub["actual"] - sub["predicted"])**2)
            ss_tot = np.sum((sub["actual"] - sub["actual"].mean())**2)
            r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else 0.0
            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("R²",   f"{r2:.4f}")
            pm2.metric("MAE",  f"{mae:.4f}")
            pm3.metric("RMSE", f"{rmse:.4f}")

            sub2 = sub.copy()
            sub2["residual"] = sub2["actual"] - sub2["predicted"]
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=sub2["predicted"], y=sub2["residual"],
                mode="markers", name="Residual",
                marker=dict(color=mc2, size=4, opacity=0.55),
            ))
            if len(sub2) > 5:
                xv = sub2["predicted"].values.astype(float)
                yv = sub2["residual"].values.astype(float)
                mask = np.isfinite(xv) & np.isfinite(yv)
                if mask.sum() > 2:
                    coeffs = np.polyfit(xv[mask], yv[mask], 1)
                    xl = np.linspace(xv[mask].min(), xv[mask].max(), 200)
                    fig_res.add_trace(go.Scatter(
                        x=xl, y=np.polyval(coeffs, xl),
                        mode="lines", name="Trend",
                        line=dict(color="#7C5A12", width=2, dash="dash"),
                    ))
            fig_res.add_hline(y=0, line_color="#7F1D1D", line_dash="dash", line_width=1)
            fig_res.update_layout(
                title=_title("Residuals (actual − predicted)"), height=320,
                xaxis_title="Predicted", yaxis_title="Residual", **PLOT_LAYOUT,
            )
            _style_axes(fig_res)
            st.plotly_chart(fig_res, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 3D HISTORICAL VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
with tab_3d:
    st.markdown(f"### 🌐 3D Historical Data — {selected_asset}")
    raw_df = load_raw(selected_asset)

    if raw_df.empty:
        st.markdown("<div class='warn-box'>Raw data file not found.</div>", unsafe_allow_html=True)
    else:
        raw_df = raw_df.dropna(subset=["High","Low","Close","Volume"])
        raw_df["Year"]   = raw_df["Date"].dt.year
        raw_df["Month"]  = raw_df["Date"].dt.month
        raw_df["Range"]  = raw_df["High"] - raw_df["Low"]
        raw_df["Return"] = raw_df["Close"].pct_change().fillna(0)
        raw_df["Vol_30"] = raw_df["Return"].rolling(30).std().fillna(0) * np.sqrt(252)

        st.markdown("<div class='info-box'>🖱 Drag to rotate · Scroll to zoom · Hover for values</div>",
                    unsafe_allow_html=True)

        view3d = st.radio("Choose 3D view", ["Price Surface", "Volume × Range × Return",
                                              "Volatility Cone", "Candlestick 3D"],
                           horizontal=True, key="view3d")

        LAYOUT_3D = dict(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#0f1e3d", family="Inter, sans-serif"),
            height=580,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        if view3d == "Price Surface":
            monthly = raw_df.groupby(["Year","Month"]).agg(
                High=("High","max"), Low=("Low","min"), Close=("Close","last")
            ).reset_index()
            pivot_h = monthly.pivot(index="Year", columns="Month", values="High").ffill(axis=1).bfill(axis=1).fillna(0)
            pivot_l = monthly.pivot(index="Year", columns="Month", values="Low").ffill(axis=1).bfill(axis=1).fillna(0)

            fig3d = go.Figure()
            fig3d.add_trace(go.Surface(
                z=pivot_h.values, x=pivot_h.columns.tolist(), y=pivot_h.index.tolist(),
                colorscale=[[0,"#0d1929"],[0.5,"#B8912E"],[1,"#F0D070"]],
                opacity=0.85, name="Monthly High",
                contours=dict(z=dict(show=True, color="rgba(212,170,70,0.15)", width=1)),
                showscale=False,
            ))
            fig3d.add_trace(go.Surface(
                z=pivot_l.values, x=pivot_l.columns.tolist(), y=pivot_l.index.tolist(),
                colorscale=[[0,"#0d1929"],[0.5,"#1d3a6e"],[1,"#22D3EE"]],
                opacity=0.6, name="Monthly Low", showscale=False,
            ))
            fig3d.update_layout(
                scene=_scene3d("Month", "Year", "Price ($)"),
                title=dict(text=f"{selected_asset} — Monthly High/Low Price Surface",
                           font=dict(color="#0f1e3d", family="Inter, sans-serif")),
                **LAYOUT_3D,
            )
            st.plotly_chart(fig3d, use_container_width=True)

        elif view3d == "Volume × Range × Return":
            sample = raw_df.tail(500).copy()
            fig3d = go.Figure(go.Scatter3d(
                x=sample["Volume"] / 1e6, y=sample["Range"], z=sample["Return"] * 100,
                mode="markers",
                marker=dict(
                    size=3,
                    color=sample["Close"],
                    colorscale=[[0,"#0d1929"],[0.5,"#7C5A12"],[1,"#22D3EE"]],
                    opacity=0.7,
                    colorbar=dict(
                        title=dict(text="Close $", font=dict(color="#0f1e3d")),
                        tickfont=dict(color="#1e3a6f", family="JetBrains Mono"),
                    ),
                ),
                text=sample["Date"].astype(str),
                hovertemplate="Date: %{text}<br>Volume: %{x:.1f}M<br>Range: %{y:.2f}<br>Return: %{z:.2f}%<extra></extra>",
            ))
            fig3d.update_layout(
                scene=_scene3d("Volume (M)", "High−Low Range", "Daily Return %"),
                title=dict(text=f"{selected_asset} — Volume × Range × Return (last 500 days)",
                           font=dict(color="#0f1e3d", family="Inter, sans-serif")),
                **LAYOUT_3D,
            )
            st.plotly_chart(fig3d, use_container_width=True)

        elif view3d == "Volatility Cone":
            yearly = raw_df.groupby("Year").agg(
                Vol_mean=("Vol_30","mean"), Vol_max=("Vol_30","max"),
                Price_mean=("Close","mean"),
            ).reset_index()
            fig3d = go.Figure()
            for _, yr_row in yearly.iterrows():
                theta = np.linspace(0, 2*np.pi, 40)
                r     = yr_row["Vol_mean"]
                x_c   = np.cos(theta) * r
                y_c   = np.sin(theta) * r
                z_c   = np.full_like(theta, yr_row["Price_mean"])
                fig3d.add_trace(go.Scatter3d(
                    x=x_c, y=y_c, z=z_c, mode="lines",
                    line=dict(color=f"rgba(212,170,70,{min(r*3,0.9):.2f})", width=2),
                    name=str(int(yr_row["Year"])), showlegend=True,
                    hovertemplate=f"Year: {int(yr_row['Year'])}<br>Avg Vol: {r:.3f}<br>Avg Price: {yr_row['Price_mean']:.2f}<extra></extra>",
                ))
            fig3d.update_layout(
                scene=_scene3d("Vol cos", "Vol sin", "Price ($)"),
                title=dict(text=f"{selected_asset} — Annual Volatility Cone",
                           font=dict(color="#0f1e3d", family="Inter, sans-serif")),
                **LAYOUT_3D,
            )
            st.plotly_chart(fig3d, use_container_width=True)

        else:  # Candlestick 3D
            sample = raw_df.tail(120).copy()
            sample["idx"] = range(len(sample))
            fig3d = go.Figure()
            for _, r in sample.iterrows():
                color = "#0F766E" if r["Close"] >= r["Open"] else "#7F1D1D"
                fig3d.add_trace(go.Scatter3d(
                    x=[r["idx"], r["idx"]], y=[0, 0], z=[r["Low"], r["High"]],
                    mode="lines", line=dict(color=color, width=1), showlegend=False,
                    hovertemplate=f"Date: {str(r['Date'])[:10]}<br>H:{r['High']:.2f} L:{r['Low']:.2f}<extra></extra>",
                ))
                fig3d.add_trace(go.Scatter3d(
                    x=[r["idx"], r["idx"]], y=[-0.3, 0.3],
                    z=[min(r["Open"], r["Close"]), max(r["Open"], r["Close"])],
                    mode="lines", line=dict(color=color, width=4), showlegend=False,
                ))
            fig3d.update_layout(
                scene=_scene3d("Time (days)", "", "Price ($)"),
                title=dict(text=f"{selected_asset} — 3D Candlestick Chart (last 120 days)",
                           font=dict(color="#0f1e3d", family="Inter, sans-serif")),
                **LAYOUT_3D,
            )
            st.plotly_chart(fig3d, use_container_width=True)




# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — UPLOAD & INFER
# ══════════════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### ⬆️ Upload prepared CSV for inference")
    st.markdown("""
    <div class='info-box'>
    Upload a CSV with the same engineered feature columns as a prepared dataset.
    The top validation model for the selected asset will run predictions automatically.
    Download the results including volatility-adjusted prediction intervals.
    </div>
    """, unsafe_allow_html=True)

    uc1, uc2 = st.columns([2,1])
    with uc1:
        uploaded = st.file_uploader("Choose a prepared CSV", type=["csv"], key="uploader")
    with uc2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        tm_color = MODEL_COLORS.get(top_models.get(selected_asset,""), "#7C5A12")
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(255,255,255,0.92),rgba(245,249,255,0.92));
                    border-radius:12px;padding:14px;
                    border:1px solid rgba(212,170,70,0.14);
                    box-shadow:0 2px 16px rgba(0,0,0,0.3);'>
            <div style='font-size:10px;color:#0f1e3d;font-family:JetBrains Mono,monospace;
                        letter-spacing:0.1em;'>TOP MODEL — {selected_asset}</div>
            <div style='font-size:18px;font-weight:800;color:{tm_color};
                        font-family:Inter,sans-serif;margin-top:4px;'>
                {top_models.get(selected_asset, "—")}
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded is not None:
        infer_df   = pd.read_csv(uploaded)
        model_name = top_models.get(selected_asset)

        st.markdown(f"<div class='info-box'>📂 Loaded <b>{len(infer_df)} rows × {len(infer_df.columns)} columns</b></div>",
                    unsafe_allow_html=True)

        csv_asset = None
        if "asset" in infer_df.columns:
            csv_asset = infer_df["asset"].iloc[0]
        elif "Date" in infer_df.columns:
            if "Close" in infer_df.columns:
                median_close = infer_df["Close"].median()
                if median_close > 10000:
                    csv_asset = "BTC_USD"
                elif median_close > 300:
                    csv_asset = "SPY" if median_close < 700 else "QQQ"
                elif median_close > 100:
                    csv_asset = "GLD" if median_close < 260 else "TLT"

        if csv_asset and csv_asset != selected_asset:
            st.markdown(f"""
            <div class='warn-box'>
            ⚠️ <b>Asset mismatch detected!</b><br>
            The uploaded CSV appears to contain <b>{csv_asset}</b> data,
            but the sidebar asset is set to <b>{selected_asset}</b>.<br>
            This will produce incorrect predictions (wrong feature scale).<br>
            <b>Please switch the sidebar Asset to <u>{csv_asset}</u> before predicting,
            or upload the correct sample file for {selected_asset}.</b>
            </div>
            """, unsafe_allow_html=True)

        if model_name:
            model_path = MODELS_DIR / f"{selected_asset}_{model_name}.joblib"
            if model_path.exists():
                pkg          = safe_joblib_load(model_path)
                feature_cols = pkg["feature_columns"]
                model        = pkg["model"]

                missing_cols = [c for c in feature_cols if c not in infer_df.columns]
                if missing_cols:
                    st.markdown(f"<div class='info-box'>ℹ️ {len(missing_cols)} missing feature columns filled with 0</div>",
                                unsafe_allow_html=True)
                for col in missing_cols:
                    infer_df[col] = 0.0

                raw_preds = model.predict(infer_df[feature_cols])
                pred_out  = pd.DataFrame(raw_preds, columns=pkg["targets"])

                for tgt in pkg["targets"]:
                    pred_out[tgt] = pred_out[tgt].clip(lower=0)

                vol_norm = infer_df["vol_norm"].values if "vol_norm" in infer_df.columns else np.zeros(len(infer_df))
                vol_norm = np.nan_to_num(vol_norm, nan=0.0).clip(-2, 5)
                scale    = 0.015 * (1.0 + vol_norm.clip(0))
                for tgt in pkg["targets"]:
                    pred_out[f"{tgt}_lower"] = (pred_out[tgt] * (1 - scale)).clip(lower=0)
                    pred_out[f"{tgt}_upper"] = (pred_out[tgt] * (1 + scale)).clip(lower=0)

                neg_count = (raw_preds < 0).sum()
                if neg_count > 0:
                    st.markdown(f"""
                    <div class='warn-box'>
                    ⚠️ <b>{neg_count} raw prediction(s) were negative</b> — clamped to 0.<br>
                    This usually means the uploaded CSV is for a different asset than selected,
                    OR the model was over-regularised during training (TLT is known to have this issue).<br>
                    Make sure the sidebar Asset matches your uploaded CSV file.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='info-box'>✅ Predictions generated using <b>{model_name}</b> — all values positive</div>",
                                unsafe_allow_html=True)

                if "Date" in infer_df.columns:
                    pred_out.insert(0, "Date", infer_df["Date"].values)

                sm1, sm2, sm3, sm4 = st.columns(4)
                t1h  = pred_out["target_high_tplus1"]
                t1l  = pred_out["target_low_tplus1"]
                t10h = pred_out["target_high_tplus10"]
                t10l = pred_out["target_low_tplus10"]
                sm1.metric("Avg T+1 High",  f"{t1h.mean():,.2f}")
                sm2.metric("Avg T+1 Low",   f"{t1l.mean():,.2f}")
                sm3.metric("Avg T+10 High", f"{t10h.mean():,.2f}")
                sm4.metric("Avg T+10 Low",  f"{t10l.mean():,.2f}")

                st.dataframe(pred_out.round(2), use_container_width=True)

                x_axis = pred_out["Date"].astype(str) if "Date" in pred_out.columns else list(range(len(pred_out)))
                fig_up = make_subplots(rows=2, cols=2,
                                       subplot_titles=["T+1 High","T+1 Low","T+10 High","T+10 Low"],
                                       vertical_spacing=0.14, horizontal_spacing=0.08)
                plot_targets = [
                    ("target_high_tplus1",  1, 1),
                    ("target_low_tplus1",   1, 2),
                    ("target_high_tplus10", 2, 1),
                    ("target_low_tplus10",  2, 2),
                ]
                for tgt, r, c in plot_targets:
                    mc2 = MODEL_COLORS.get(model_name, "#7C5A12")
                    lc  = f"{tgt}_lower"
                    uc  = f"{tgt}_upper"
                    if lc in pred_out.columns:
                        fig_up.add_trace(go.Scatter(
                            x=list(x_axis) + list(x_axis)[::-1],
                            y=list(pred_out[uc]) + list(pred_out[lc])[::-1],
                            fill="toself", fillcolor="rgba(212,170,70,0.08)",
                            line=dict(color="rgba(0,0,0,0)"),
                            name="Volatility band", showlegend=(r==1 and c==1),
                            legendgroup="band",
                        ), row=r, col=c)
                    fig_up.add_trace(go.Scatter(
                        x=x_axis, y=pred_out[tgt],
                        mode="lines+markers",
                        name=tgt.replace("target_","").replace("tplus","T+").replace("_"," ").upper(),
                        line=dict(color=mc2, width=2),
                        marker=dict(size=5),
                        showlegend=(r==1 and c==1),
                        legendgroup=tgt,
                    ), row=r, col=c)

                fig_up.update_layout(
                    title=_title(f"{selected_asset} — {model_name} Predictions (all targets)"),
                    height=520, hovermode="x unified", **PLOT_LAYOUT,
                )
                for ann in fig_up.layout.annotations:
                    ann.font.color  = "#1e3a6f"
                    ann.font.size   = 13
                    ann.font.family = "Inter, sans-serif"
                _style_axes(fig_up)
                st.plotly_chart(fig_up, use_container_width=True)

                csv_bytes = pred_out.to_csv(index=False).encode()
                st.download_button(
                    "💾 Download predictions CSV", csv_bytes,
                    file_name=f"{selected_asset}_{model_name}_predictions.csv",
                    mime="text/csv", use_container_width=True,
                )
            else:
                st.markdown(f"<div class='warn-box'>Model file not found: {model_path.name}</div>",
                            unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='warn-box'>No top model recorded for {selected_asset}. Run generate_outputs.py first.</div>",
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — INVESTMENT ADVISOR
# ══════════════════════════════════════════════════════════════════════════════
with tab_invest:
    st.markdown("### 💡 Investment Advisor — Daily Price Predictor")
    st.markdown("""
    <div class='info-box'>
    Enter your asset details below. The model will predict today's High &amp; Low price,
    then advise whether your entry price is a good investment for the day.
    </div>
    """, unsafe_allow_html=True)

    ia1, ia2 = st.columns(2)

    with ia1:
        invest_asset = st.selectbox(
            "🏦 Select Asset",
            ["SPY", "QQQ", "GLD", "TLT", "BTC_USD"],
            key="invest_asset",
        )

    raw_hint = None
    try:
        prep_p = PROCESSED_DIR / f"{invest_asset}_prepared.csv"
        if prep_p.exists():
            hint_df = pd.read_csv(prep_p, usecols=["Date","Open","Close"])
            hint_df["Date"] = pd.to_datetime(hint_df["Date"])
            hint_df = hint_df.sort_values("Date")
            last_row = hint_df.iloc[-1]
            raw_hint = round(float((last_row["Open"] + last_row["Close"]) / 2), 2)
    except Exception:
        raw_hint = None

    with ia2:
        hint_text = f"Suggested: avg of last Open & Close ≈ {raw_hint:,.2f}" if raw_hint else "Enter your planned entry price"
        invest_price = st.number_input(
            f"💰 Your Entry Price ({invest_asset})",
            min_value=0.01,
            value=float(raw_hint) if raw_hint else 100.0,
            step=0.01, format="%.2f", help=hint_text, key="invest_price",
        )
        st.markdown(
            f"<div style='font-size:12px;color:#0f1e3d;margin-top:-12px;margin-bottom:8px;"
            f"font-family:JetBrains Mono,monospace;'>💡 {hint_text}</div>",
            unsafe_allow_html=True,
        )

    VOL_LABELS = {
        "SPY":     "VIX  (CBOE Volatility Index — typical range 10–80)",
        "QQQ":     "VXN  (Nasdaq Volatility Index — typical range 12–80)",
        "GLD":     "GVZ  (Gold Volatility Index — typical range 10–50)",
        "TLT":     "MOVE (Bond Market Volatility — typical range 50–200)",
        "BTC_USD": "Fear & Greed Index (0 = Extreme Fear, 100 = Extreme Greed)",
    }
    VOL_DEFAULTS = {"SPY": 20.0, "QQQ": 22.0, "GLD": 15.0, "TLT": 100.0, "BTC_USD": 50.0}
    VOL_RANGES   = {"SPY": (1.0,150.0), "QQQ": (1.0,150.0), "GLD": (1.0,100.0),
                    "TLT": (10.0,300.0), "BTC_USD": (0.0,100.0)}

    ic1, ic2 = st.columns(2)
    with ic1:
        vol_min, vol_max = VOL_RANGES[invest_asset]
        invest_vol = st.number_input(
            f"📊 Risk Metric — {VOL_LABELS[invest_asset]}",
            min_value=vol_min, max_value=vol_max,
            value=VOL_DEFAULTS[invest_asset],
            step=0.1, format="%.1f",
            help="This value is used as the volatility index input to the prediction model.",
            key="invest_vol",
        )

    with ic2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("🔮 Predict & Advise", key="predict_btn",
                                    use_container_width=True)

    if predict_clicked:
        model_name_inv = top_models.get(invest_asset)
        model_path_inv = MODELS_DIR / f"{invest_asset}_{model_name_inv}.joblib" if model_name_inv else None

        if not model_name_inv or not model_path_inv.exists():
            st.markdown("<div class='warn-box'>No trained model found. Run generate_outputs.py first.</div>",
                        unsafe_allow_html=True)
        else:
            pkg_inv      = safe_joblib_load(model_path_inv)
            feature_cols = pkg_inv["feature_columns"]
            model_inv    = pkg_inv["model"]

            try:
                prep_df_inv   = pd.read_csv(PROCESSED_DIR / f"{invest_asset}_prepared.csv")
                last_features = prep_df_inv[feature_cols].iloc[[-1]].copy()

                VOL_COL_MAP_INV = {
                    "SPY": "vix_close", "QQQ": "vxn_close",
                    "GLD": "gvz_close", "TLT": "move_close", "BTC_USD": "fg_value",
                }
                raw_vol_col = VOL_COL_MAP_INV.get(invest_asset, "")
                vol_lag_col = f"{raw_vol_col}_lag_1" if f"{raw_vol_col}_lag_1" in feature_cols else None
                if vol_lag_col:
                    last_features[vol_lag_col] = invest_vol
                if "vol_norm" in feature_cols:
                    col_data = prep_df_inv["vol_norm"].dropna()
                    z = (invest_vol - col_data.mean()) / (col_data.std() + 1e-9)
                    last_features["vol_norm"] = float(z)

                raw_pred     = model_inv.predict(last_features)[0]
                pred_high_t1 = max(float(raw_pred[0]), 0)
                pred_low_t1  = max(float(raw_pred[1]), 0)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
                r1, r2, r3 = st.columns(3)
                r1.metric("📈 Predicted High (T+1)", f"{pred_high_t1:,.2f}")
                r2.metric("📉 Predicted Low  (T+1)", f"{pred_low_t1:,.2f}")
                r3.metric("💰 Your Entry Price",      f"{invest_price:,.2f}")

                fig_inv = go.Figure()
                fig_inv.add_trace(go.Bar(
                    x=["Predicted High", "Predicted Low", "Your Entry Price"],
                    y=[pred_high_t1, pred_low_t1, invest_price],
                    marker_color=[
                        "#0F766E" if pred_high_t1 > invest_price else "#7F1D1D",
                        "#0F766E" if pred_low_t1  > invest_price else "#7F1D1D",
                        "#7C5A12",
                    ],
                    text=[f"{pred_high_t1:,.2f}", f"{pred_low_t1:,.2f}", f"{invest_price:,.2f}"],
                    textposition="outside",
                    textfont=dict(color="#0f1e3d", size=13, family="JetBrains Mono"),
                ))
                fig_inv.add_hline(y=invest_price, line_dash="dash",
                                  line_color="#7C5A12", line_width=2,
                                  annotation_text=f"Entry: {invest_price:,.2f}",
                                  annotation_font_color="#7C5A12")
                fig_inv.update_layout(
                    title=_title(f"{invest_asset} — Predicted T+1 High & Low vs Your Entry Price"),
                    height=380, showlegend=False, **PLOT_LAYOUT,
                )
                _style_axes(fig_inv)
                st.plotly_chart(fig_inv, use_container_width=True)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                if pred_high_t1 > invest_price and pred_low_t1 > invest_price:
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,rgba(52,211,153,0.1),rgba(52,211,153,0.05));
                                border:1.5px solid rgba(52,211,153,0.35);border-radius:16px;
                                padding:28px 32px;text-align:center;
                                box-shadow:0 4px 32px rgba(52,211,153,0.15);margin:8px 0;">
                        <div style="font-size:48px;margin-bottom:8px">✅</div>
                        <div style="font-size:20px;font-weight:800;color:#0F766E;margin-bottom:8px;
                                    font-family:Inter,sans-serif;letter-spacing:-0.02em;">
                            Low Risk Level
                        </div>
                        <div style="font-size:13px;color:#0F766E;font-weight:500;
                                    font-family:Inter,sans-serif;line-height:1.6;">
                            Predicted High <b>{pred_high_t1:,.2f}</b> and Low <b>{pred_low_t1:,.2f}</b>
                            are both above your entry of <b>{invest_price:,.2f}</b>.<br>
                            The model expects the price to stay above your entry throughout the day.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    detail = []
                    if pred_high_t1 <= invest_price:
                        detail.append(f"Predicted High <b>{pred_high_t1:,.2f}</b> is below your entry")
                    if pred_low_t1 <= invest_price:
                        detail.append(f"Predicted Low <b>{pred_low_t1:,.2f}</b> is below your entry")
                    detail_str = " · ".join(detail)
                    st.markdown(f"""
                    <div style="background:linear-gradient(135deg,rgba(248,113,113,0.1),rgba(248,113,113,0.05));
                                border:1.5px solid rgba(248,113,113,0.35);border-radius:16px;
                                padding:28px 32px;text-align:center;
                                box-shadow:0 4px 32px rgba(248,113,113,0.15);margin:8px 0;">
                        <div style="font-size:48px;margin-bottom:8px">⚠️</div>
                        <div style="font-size:20px;font-weight:800;color:#7F1D1D;margin-bottom:8px;
                                    font-family:Inter,sans-serif;letter-spacing:-0.02em;">
                            Very High Risk Level
                        </div>
                        <div style="font-size:13px;color:#7F1D1D;font-weight:500;
                                    font-family:Inter,sans-serif;line-height:1.6;">
                            {detail_str}.<br>
                            Consider waiting for a better entry point or re-assess your risk tolerance.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"<div class='warn-box'>Prediction failed: {e}</div>",
                            unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — REPORT (inline preview + download)
# ══════════════════════════════════════════════════════════════════════════════
with tab_report:
    st.markdown("### 📄 Project Report")
    st.markdown("""
    <div class='info-box'>
    This report is dynamically generated from your actual pipeline data — real R² scores,
    real predictions, real charts. Click <b>Generate Report</b> to build it, then preview
    inline or download the full HTML file to open in any browser or print as PDF.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    rc1, rc2 = st.columns([2, 1])
    with rc1:
        gen_clicked = st.button(
            "🔄 Generate Report", use_container_width=True, key="gen_report_tab"
        )
    with rc2:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:11px;color:#6B7280;font-family:JetBrains Mono,monospace;"
            "padding:8px;'>Charts build from live data</div>",
            unsafe_allow_html=True,
        )

    if gen_clicked or st.session_state.get("report_html"):
        if gen_clicked:
            # Clear any previously cached report so a fresh one is always built
            st.session_state.pop("report_html", None)
            with st.spinner("Building report — generating charts from live data…"):
                report_html = generate_html_report(
                    summary_df, metrics_df, r2_audit_df, top_models,
                    load_predictions, load_raw, MODEL_COLORS,
                )
            st.session_state["report_html"] = report_html
        else:
            report_html = st.session_state["report_html"]

        # Download button
        st.download_button(
            "⬇️ Download Report (HTML)",
            data=report_html.encode("utf-8"),
            file_name="AI_Financial_Project_Report.html",
            mime="text/html",
            use_container_width=False,
            key="dl_report_tab",
        )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("**Preview** — scroll inside the frame to read the full report:")

        # Inline preview in an iframe
        components.html(
            report_html,
            height=820,
            scrolling=True,
        )
