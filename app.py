import streamlit as st
import pandas as pd

# Set page config
st.set_page_config(page_title="UAC System Capacity Analytics", layout="wide")

# Load Data
@st.cache_data
def load_data():
    # Load without index first to verify column names, then set index
    data = pd.read_csv('processed_uac_data.csv')
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
    return data

# Raw data load
df_raw = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters")

# Date Range Selector
if not df_raw.empty:
    min_date = df_raw.index.min().to_pydatetime()
    max_date = df_raw.index.max().to_pydatetime()
    start_date, end_date = st.sidebar.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )

    # Time Granularity Selector
    granularity = st.sidebar.selectbox("Time Granularity", ["Daily", "Weekly", "Monthly"])
    gran_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}

    # Filter and Resample Data
    df_filtered = df_raw.loc[start_date:end_date].copy()
    if granularity != "Daily":
        df_display = df_filtered.resample(gran_map[granularity]).sum(numeric_only=True)
    else:
        df_display = df_filtered

    # Metric Toggles
    show_cbp = st.sidebar.checkbox("Show CBP Load", value=True)
    show_hhs = st.sidebar.checkbox("Show HHS Load", value=True)

    st.title("System Capacity & Care Load Analytics for HHS")

    # --- KPI SUMMARY CARDS ---
    current_total_care = df_filtered['Total System Load'].iloc[-1]
    avg_total_care = df_filtered['Total System Load'].mean()
    current_net_intake = df_filtered['Net Daily Intake'].iloc[-1]
    backlog_30d = df_filtered['Net Daily Intake'].tail(30).mean()

    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Children Under Care", f"{current_total_care:,.0f}")
    with col2:
        st.metric("Avg System Responsibility", f"{avg_total_care:,.0f}")
    with col3:
        st.metric("Current Net Intake", f"{current_net_intake:,.0f}")
    with col4:
        st.metric("30d Backlog Accumulation", f"{backlog_30d:.2f}")

    st.divider()

    # --- DASHBOARD MODULES ---

    # 1. System Load Overview
    st.subheader("System Load Overview Pane")
    st.line_chart(df_display['Total System Load'])

    # 2. CBP vs HHS Load Comparison
    st.subheader("CBP vs HHS Load Comparison")
    comparison_cols = []
    if show_cbp: comparison_cols.append('Children in CBP custody')
    if show_hhs: comparison_cols.append('Children in HHS Care')

    if comparison_cols:
        st.area_chart(df_display[comparison_cols])
    else:
        st.warning("Please select at least one metric in the sidebar to view comparison.")

    # 3. Net Intake & Backlog Trends
    st.subheader("Net Intake & Backlog Trends")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("**Net Daily Intake (Flow)**")
        st.bar_chart(df_display['Net Daily Intake'])

    with chart_col2:
        st.write("**Load Volatility (Rolling Std Dev)**")
        st.line_chart(df_display['7-Day Rolling Std Dev Load'])
else:
    st.error("Data could not be loaded. Please check the CSV file.")
