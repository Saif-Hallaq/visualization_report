# Refactored version of the long Streamlit dashboard
# Keeping all logic, filters, and features intact, but organized and modular.

import time
import streamlit as st
import pandas as pd
import plotly.express as px
import io
from helpers import (
    load_data,
    create_pie_chart,
    filter_by_timeframe,
    render_tab_analysis,
    render_tab_overview,
    render_tab_sources,
    render_tab_datenblatt
)

# Page title
st.set_page_config(page_title="📊 Excel Data Dashboard", layout="wide")

# Sidebar navigation
tab = st.sidebar.radio(
    "",
    [
        "Übersicht",
        "Auswertungen nach Suchagenten",
        "Auswertungen nach Tags",
        "Auswertungen nach Smart-Tags",
        "Auswertungen nach Quellen",
        "Datenblatt"
    ]
)

# Custom CSS to modify the file uploader text
custom_css = """
    <style>
        div[data-testid="stFileUploaderDropzoneInstructions"] > div > span {
            display: none;
        }
        div[data-testid="stFileUploaderDropzone"] > button {
            color: transparent;
            position: relative;
        }
        div[data-testid="stFileUploaderDropzone"] > button::after {
            content: "📂 Datei auswählen";
            color: black;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 16px;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] > div > small {
            visibility: hidden;
        }
        div[data-testid="stFileUploaderDropzoneInstructions"] > div > small::before {
            content: "Limit: 200 MB pro Datei • XLS, XLSX";
            visibility: visible;
            display: block;
            color: #555;
            font-size: 12px;
            margin-top: 5px;
        }
    </style>
"""

# Apply the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

# File uploader
with st.sidebar:
    uploaded_file = st.file_uploader("Datei hochladen", type=["xls", "xlsx"])

# Load and cache data
if "df" not in st.session_state or uploaded_file:
    st.session_state.df = load_data(uploaded_file) if uploaded_file else None

df = st.session_state.df

# Proceed with rendering each tab if data exists
if df is not None:
    if tab == "Übersicht":
        render_tab_overview(df)
    elif tab == "Auswertungen nach Suchagenten":
        render_tab_analysis(df, mode="Suchagent")
    elif tab == "Auswertungen nach Tags":
        render_tab_analysis(df, mode="Tag")
    elif tab == "Auswertungen nach Smart-Tags":
        render_tab_analysis(df, mode="Smart-Tag")
    elif tab == "Auswertungen nach Quellen":
        render_tab_sources(df)
    elif tab == "Datenblatt":
        render_tab_datenblatt(df)
else:
    st.warning("⚠️ Bitte laden Sie eine Excel-Datei hoch, um zu beginnen.")
