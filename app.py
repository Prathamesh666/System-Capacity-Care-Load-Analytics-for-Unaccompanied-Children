import streamlit as st # type: ignore
st.set_page_config(
    page_title="System Capacity & Care Load Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
from model import train_and_evaluate_models, train_and_evaluate_classifiers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings 
warnings.filterwarnings("ignore")

# -------------------------
# Helpers & caching
# -------------------------
@st.cache_data(ttl=3600)
def load_and_prepare(source):
    df = pd.read_csv(source)
    if 'Date' not in df.columns:
        raise ValueError("CSV must contain 'Date' column.")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    numeric_cols = [
        'Children apprehended and placed in CBP custody*',
        'Children in CBP custody',
        'Children transferred out of CBP custody',
        'Children in HHS Care',
        'Children discharged from HHS Care'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
    df = df.sort_values('Date')
    if not df['Date'].is_unique:
        df = df.groupby('Date').sum(numeric_only=True).reset_index()
    df = df.set_index('Date')
    full_range = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(full_range).fillna(0)
    df.index.name = 'Date'

    # Derived metrics
    df['Total System Load'] = df.get('Children in CBP custody', 0) + df.get('Children in HHS Care', 0)
    df['Net Daily Intake'] = df.get('Children transferred out of CBP custody', 0) - df.get('Children discharged from HHS Care', 0)
    df['Care Load Growth Rate'] = df['Total System Load'].pct_change() * 100
    df['Care Load Growth Rate'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Positive Net Intake'] = (df['Net Daily Intake'] > 0).astype(int)
    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month
    return df

# --- CSS for metric styling with hover + bouncing animation ---
st.markdown("""
    <style>
    /* Center text inside the alert */
    .stAlert {
        text-align: center;
    }
    
    /* Base metric card */
    div[data-testid="stMetric"] { background-color: rgba(255,255,255,0.05); border-radius: 12px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.15); transition: all 0.3s ease; text-align: left; overflow: hidden; position: relative;
    }

    /* Hover effect */
    div[data-testid="stMetric"]:hover { box-shadow: 0 6px 14px rgba(0,0,0,0.25); transform: translateY(-4px); background-color: #FFD700;
    }

    /* Metric label */
    div[data-testid="stMetric"] > label { font-size: 0.9rem; font-weight: 600; color: #cccccc;
    }

    /* Metric value with bouncing animation */
    div[data-testid="stMetric"] > div {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.1rem;
        display: inline-block;
        white-space: nowrap;
        animation: bounceXY 6s ease-in-out infinite alternate;
    }

    /* Delta styling with bounce */
    div[data-testid="stMetricDelta"] {
        font-size: 0.9rem;
        font-weight: 600;
        animation: bounceXY 9s ease-in-out infinite alternate;
    }

    /* Bouncing animation across both axes */
    @keyframes bounceXY {
        0%   { transform: translate(0%, 0%); }
        33%  { transform: translate(-120%, 0%); }
        67%  { transform: translate(120%, 0%); }
        100% { transform: translate(0%, 0%); }
    }
    </style>
""", unsafe_allow_html=True)
st.logo("static/Banner_Symbol.png")
# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.image(
    "static/Banner.png",
    width='stretch'
)
st.sidebar.title("Controls")

# --- Sidebar Upload Option ---
uploaded_file = st.sidebar.file_uploader(
    "Upload your Custom CSV file (optional)",
    type=["csv"]
)

# Required schema for HHS dataset
required_columns = [ "Date", "Children apprehended and placed in CBP custody*", "Children in CBP custody",
    "Children transferred out of CBP custody", "Children in HHS Care", "Children discharged from HHS Care" ]

# -------------------------
# Load data with preprocessing
# -------------------------
try:
    if uploaded_file is not None:
        df_user = pd.read_csv(uploaded_file)

        # Validate schema
        if all(col in df_user.columns for col in required_columns):
            st.success("✅ Custom HHS CSV uploaded successfully!")
            df_hhs = load_and_prepare(uploaded_file)  # use your helper
        else:
            st.error("❌ Uploaded CSV does not satisfy analytical requirements.")
            with st.expander("Required Columns"):
                st.write(required_columns)
            # Fallback to default dataset
            df_hhs = load_and_prepare("HHS_Unaccompanied_Alien_Children_Program.csv")
    else:
        # Default dataset
        df_hhs = load_and_prepare("HHS_Unaccompanied_Alien_Children_Program.csv")

    # --- Date Range Slider ---
    try:
        min_date = df_hhs.index.min()
        max_date = df_hhs.index.max()

        # Slider for selecting date range
        date_range = st.sidebar.slider(
            "Select Date Range",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime())
        )

        # Apply filter
        df_hhs = df_hhs.loc[date_range[0]:date_range[1]]

    except Exception as e:
        st.warning(f"⚠️ Date filter could not be applied: {e}")

except Exception as e:
    st.error(f"⚠️ Error loading dataset: {e}")
    st.info("Falling back to default dataset.")
    try:
        df_hhs = load_and_prepare("HHS_Unaccompanied_Alien_Children_Program.csv")
        st.caption("📊 Default HHS dataset loaded successfully.")
    except Exception as e2:
        st.error(f"❌ Failed to load default dataset: {e2}")
        df_hhs = pd.DataFrame()  # empty fallback

# Final assignment for downstream use
try:
    df = df_hhs.copy()
except Exception as e:
    st.error(f"Failed to load data: {e}")

# -------------------------
# Page header
# -------------------------
st.title("📊 From Border to Care: CBP–HHS Custody-to-Care Analytics")
st.snow()   # ❄️ Snow breeze effect

# Greeting
st.caption("### 👋 Welcome! To the **From Border to Care: CBP–HHS Custody-to-Care Analytics Dashboard**.")
    
# CBP Section
col1, col2 = st.columns([1, 5])

with col1:
    st.image("static/CBP.png", width='stretch')  # display the CBP logo

with col2:
    st.markdown("#### 🏢 About CBP")
    st.caption(
        "The **U.S. Customs and Border Protection (CBP)** is the frontline agency responsible for apprehending unaccompanied children at the border. "
        "CBP agents process these children, determine their legal status, and ensure they are safely transferred to appropriate care facilities. "
        "Learn more: [CBP Official Site](https://www.cbp.gov)"
    )

# HHS Section
col1, col2 = st.columns([5, 1])

with col2:
    st.image("static/HHS.png", width='stretch')  # display the CBP logo

with col1:
    st.markdown("#### 🏥 About HHS")
    st.caption(
        "The **U.S. Department of Health and Human Services (HHS)**, through its **Office of Refugee Resettlement (ORR): A Parent Agency Administration for Children and Families (ACF)**, provides shelter, healthcare, education, and case management for unaccompanied children once they are referred from CBP. "
        "HHS ensures safe placements, reunification with sponsors, and ongoing support services. "
        "Learn more: [HHS.gov](https://www.hhs.gov)"
    )

# Dashboard Structure
st.markdown("#### 🔹 Dashboard Structure")
st.caption(
    "- **Structural Forecast** — descriptive & diagnostic insights into system load, custody transfers, healthcare metrics, and stress indicators.\n"
    "- **Recommendational Forecast** — feature engineering, predictive modeling, and actionable recommendations.\n\n"
    "Navigate through the tabs to explore visualizations, KPIs, and contextual insights designed for clarity and impact."
)

tab_struct, tab_reco = st.tabs(["🏗️ Structural Forecast", "🎯 Recommendational Forecast"])

# -------------------------
# Structural Forecast Tab
# -------------------------
with tab_struct:
    st.header("🏗️ Structural Forecast — System Insights")
    st.markdown(
        """
        This section provides a **comprehensive diagnostic view** of system capacity and care load.    

        🔹 **Purpose**: Highlight descriptive trends, stress indicators, and healthcare metrics.  
        🔹 **Format**: Subtabs group related visualizations, while expanders provide concise explanations.  
        🔹 **Interactivity**: Toggle buttons, plots and hover effects enhance engagement without overwhelming the user."""
    )

    st.info("Navigate through subtabs to explore ingestion checks, individual trends, derived healthcare KPIs, temporal analyses, and stress indicators.")

    # Top KPIs
    st.markdown("### 📊 Top Key Performance Indicators (KPIs)")
    k1, k2, k3, k4 = st.columns(4)
    
    # First row of KPIs
    k1.metric("👶 Total Children Under Care", f"{int(df['Total System Load'].iloc[-1]):,}")
    k2.metric("📊 Avg System-wide Responsibility", f"{int(df['Total System Load'].mean()):,}")
    k3.metric("➕ Positive Net Intake Days", int(df['Positive Net Intake'].sum()))
    k4.metric("📈 90th Percentile Load", f"{int(df['Total System Load'].quantile(0.90)):,}")
    
    # Discharge Offset Ratio calculation
    df['Discharge Offset Ratio'] = df['Children discharged from HHS Care'] / df['Children transferred out of CBP custody'].replace(0, np.nan)
    df['Discharge Offset Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Discharge Offset Ratio'].fillna(0, inplace=True)
    
    # Second row of KPIs
    k1.metric("🔄 Discharge Offset Ratio", f"{df['Discharge Offset Ratio'].mean():.2f}")
    k2.metric("⚡ Net Intake Pressure", f"{int(df['Net Daily Intake'].iloc[-1]):,}")
    k3.metric("📉 Care Load Volatility Index", f"{df['Care Load Growth Rate'].std():.2f}%")
    k4.metric("⏳ Backlog Accumulation Rate", f"{df['Positive Net Intake'].rolling(window=7).mean().iloc[-1]:.4f} (last 7 days)")
    
    # Structural subtabs
    s_ingest, s_indiv, s_kpi, s_trend, s_pressure = st.tabs([
        "📥 Data Ingestion & Validation", "📈 Individual Trends", "⚕️ Derived Healthcare Metrics", "⏳ Trend & Temporal Analysis", "⚠️ Pressure & Stress"
    ])

    # -------------------------
    # Subtab: Data Ingestion & Validation
    # -------------------------
    with s_ingest:
        st.subheader("📥 Data Ingestion & ✅ Validation")
    
        with st.expander("👀 Demo Dataset Preview: Reproduces ingestion, cleaning, and validation steps from the notebook."):
            st.dataframe(df.head(10), width='stretch')
    
        missing_dates = df.index[df.isnull().all(axis=1)].tolist()
        duplicated_dates = df.index[df.index.duplicated()].tolist()
        c1, c2, c3 = st.columns(3)
        c1.metric("📅 Total Days", len(df))
        c2.metric("⚠️ Missing Dates", len(missing_dates))
        c3.metric("🔁 Duplicate Dates", len(duplicated_dates)) 
    
        with st.expander("ℹ️ More Information"):
            st.info("🗓️ Data reindexed to a complete daily range; missing days filled with zeros.")
            st.success("🧹 Cleaning applied: numeric coercion, duplicate aggregation, chronological ordering.")
    
        # Constraint checks
        c_transfers = df[df.get('Children transferred out of CBP custody', 0) > df.get('Children in CBP custody', 0)]
        c_discharges = df[df.get('Children discharged from HHS Care', 0) > df.get('Children in HHS Care', 0)]
        if not c_transfers.empty or not c_discharges.empty:
            with st.expander("🚨 Reporting anomalies (detailed)"):
                st.warning("⚠️ Anomalies detected where transfers/discharges exceed custody/care.")
                if not c_transfers.empty:
                    st.info("🔄 Transfers > CBP custody (sample):")
                    st.dataframe(c_transfers[['Children transferred out of CBP custody', 'Children in CBP custody']].head(10))
                if not c_discharges.empty:
                    st.info("🏥 Discharges > HHS care (sample):")
                    st.dataframe(c_discharges[['Children discharged from HHS Care', 'Children in HHS Care']].head(10))
                st.info("🛠️ Action: verify source reporting or apply business rules to cap flows.")
        else:
            st.success("✅ No reporting anomalies detected.")
    
    # -------------------------
    # Subtab: Individual Trends
    # -------------------------
    with s_indiv:
        st.subheader("📊 Individual Trends: Daily Metrics")
        st.info("⚕️ Daily-Flow dynamic lanes of children in CPB custody & HHS care")
    
        children_metrics = [
            'Children apprehended and placed in CBP custody*',
            'Children in CBP custody',
            'Children transferred out of CBP custody',
            'Children in HHS Care',
            'Children discharged from HHS Care'
        ]
        # 3D consolidated flow (lanes)
        st.markdown("#### 🌐 Metric Lanes")
    
        labels = ['🧒 Apprehended', '📦 In CBP', '🔄 Transferred out', '🏥 In HHS', '✅ Discharged']
        fig3d = go.Figure()
        colors = px.colors.qualitative.Bold
    
        for i, m in enumerate(children_metrics):
            if m in df.columns:
                plot_df = df[[m]].dropna()
                fig3d.add_trace(go.Scatter3d(
                    x=plot_df.index,
                    y=[i] * len(plot_df),
                    z=plot_df[m],
                    mode='lines',
                    name=labels[i],  # emoji-enhanced labels
                    line=dict(width=4, color=colors[i % len(colors)])
                ))
    
        fig3d.update_layout( 
            title=dict(text="🌐 <b>3D Consolidated Flow of Metrics</b>", x=0.5, xanchor='center', yanchor='top'),
            scene=dict( xaxis=dict(title=dict(text="📅 Date")), yaxis=dict(title=dict(text="📊 Metric Segment")), 
                    zaxis=dict(title=dict(text="🔢 Count")), aspectmode="cube", camera=dict( eye=dict(x=1.5, y=1.5, z=1.2))),
            legend=dict(title="📌 Metrics", orientation="h", yanchor="bottom", y=-0.4, x=0.5, xanchor="center"), 
            margin=dict(l=0, r=0, t=50, b=80), autosize=True
        )
    
        st.plotly_chart(fig3d, width='stretch')
        
        with st.expander("ℹ️ More Information"):
            st.info("🌀 3D lanes reduce overlap and reveal temporal correlations across metrics.")
            st.success("✨ Each lane is symbol-coded for quick recognition (e.g., 🧒 Apprehended, 🏥 HHS Care).")
        
        metric_choice = st.selectbox("🔎 Choose metric to view", children_metrics, index=1)
        fig_metric = px.line(df, x=df.index, y=metric_choice, title=f"📈 Daily Trend: {metric_choice}")
        fig_metric.update_traces(line=dict(color='teal'))
        fig_metric.update_layout(
            title=dict( text=f"📈 Daily Trend: {metric_choice}", x=0.5, xanchor="center", yanchor="top" ),
            xaxis=dict( title="📅 Date", automargin=True ),
            yaxis=dict( title=f"👶 {metric_choice.split()[0]}",  title_standoff=10, automargin=True ),
            autosize=True, margin=dict(l=40, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_metric, width='stretch')
    
        with st.expander("ℹ️ More Information"):
            st.info("📆 Daily trend for the selected metric after cleaning.")
            st.success("🖼️ Snapshot of small multiples below show all metrics for quick comparison.")
    
        # Small multiples
        with st.expander("🖼️ Snapshot of All Metrics"):
            cols = st.columns(3)  # 3 charts per row
            for i, m in enumerate(children_metrics):
                # Split metric into words
                words = m.split()
                title_text = m
                fig = px.line(df, x=df.index, y=m)
                fig.update_layout(
                    title=dict(text=f"📊 {title_text}",  font=dict(size=10), x=0.5, xanchor="center"),
                    yaxis=dict(title=f"👶 {m.split()[0]}", title_standoff=10, automargin=True),
                    xaxis=dict( title="📅 Date", automargin=True ), autosize=True, margin=dict(l=20, r=20, t=50, b=50)
                )
                cols[i % 3].plotly_chart(fig, width='stretch')
        
        st.markdown("**🔍 Individual Trend Insights**")
        s1, s2, s3 = st.columns(3)
        s1.metric("📌 Peak CBP Custody", f"{int(df['Children in CBP custody'].max()):,}")
        s2.metric("🏥 Peak HHS Care", f"{int(df['Children in HHS Care'].max()):,}")
        discharge_offset = (df['Children discharged from HHS Care'].sum() / max(1, df['Children transferred out of CBP custody'].sum()))
        s3.metric("🔄 Discharge Offset Ration of Total", f"{discharge_offset:.2f}")

    # -------------------------
    # Subtab: Derived Healthcare Metrics
    # -------------------------
    with s_kpi:
        st.subheader("Derived Healthcare Capacity Metrics")
        st.info("🧮 Total System Load, Net Daily Intake, Growth Rate, Discharge Effectiveness, KPI surfaces, and the two KPI discharge graphs.")
        
        st.markdown("### ⚖️ Daily Flow Dynamics (CBP + HHS)")
        st.info("📊 Total System Load = CBP custody + HHS care.")
        st.success("👶 Represents the total number of children under care on any given day.")
        st.warning("⚠️ Sudden spikes may indicate operational strain or reporting anomalies.")
        try:
            # 1. Total System Load
            fig_daily_CPB_Plus_HHS = px.line(
                df,
                x=df.index,
                y='Total System Load',
                labels={'Total System Load': '👶 Total Children Under Care', 'index': '📅 Date'}
            )
            fig_daily_CPB_Plus_HHS.update_layout(title=dict( text="📈 <b>Daily Total System Load (CBP + HHS)</b>", x=0.5, xanchor='center', yanchor='top' ),
            autosize=True )
            st.plotly_chart(fig_daily_CPB_Plus_HHS, width='stretch')
        
            with st.expander("ℹ️ More Information about Daily Load"):
                avg_load = df['Total System Load'].mean()
                max_load = df['Total System Load'].max()
                busiest_day = df['Total System Load'].idxmax().date()
                latest_date = df.index.max()
                latest_val = df.loc[latest_date, 'Total System Load']
                st.success(f"✅ Average daily load: {avg_load:,.0f}. Peak load: {max_load:,.0f} on 📅 {busiest_day}.")
                st.warning(f"⚠️ Latest value on 📅 {latest_date.date()} is {latest_val:,.0f}.")
        
            st.markdown("### 📊 Net Intake & Care Load Growth (%)")
            
            col1, col2 = st.columns(2)
            
            # 2. Net Daily Intake
            fig_net_intake = px.line(
                df,
                x=df.index,
                y='Net Daily Intake',
                labels={'Net Daily Intake': '📥 Net Children Intake', 'index': '📅 Date'}
            )
            fig_net_intake.update_layout( title=dict( text="📈 <b>Net Daily Intake</b>", x=0.5, xanchor='center', yanchor='top' ), autosize=True )
            with col1:
                st.info("📊 Net Daily Intake = Transfers out of CBP - Discharges from HHS.")
                st.success("🔄 Positive values indicate net inflow; negative values indicate net outflow.")
                st.warning("⚠️ Sustained positive net intake may signal growing operational pressure; sustained negative may indicate relief or issues.")
                st.plotly_chart(fig_net_intake, width='stretch')
            
                with st.expander("ℹ️ More Information about Net Intake"):
                    avg_intake = df['Net Daily Intake'].mean()
                    max_intake = df['Net Daily Intake'].max()
                    min_intake = df['Net Daily Intake'].min()
                    st.info(f"ℹ️ Average net intake: {avg_intake:,.0f}.")
                    st.success(f"📈 Maximum intake: {max_intake:,.0f}.")
                    st.error(f"📉 Minimum intake: {min_intake:,.0f}.")
        
            # 3. Daily Care Load Growth Rate
            fig_growth_rate = px.line(
                df,
                x=df.index,
                y='Care Load Growth Rate',
                labels={'Care Load Growth Rate': '📊 Growth Rate (%)', 'index': '📅 Date'}
            )
            fig_growth_rate.update_layout( title=dict( text="📈 <b>Daily Care Load Growth Rate (%)</b>", x=0.5, xanchor='center', yanchor='top' ),
            autosize=True )
            with col2:
                st.info("📊 Care Load Growth Rate = % change in Total System Load from the previous day.")
                st.success("📈 Positive values indicate growth; negative values indicate decline.")
                st.warning("⚠️ Sudden spikes in growth rate may indicate stress windows that require closer monitoring.")
                st.plotly_chart(fig_growth_rate, width='stretch')
            
                with st.expander("ℹ️ More Information about Growth Rate"):
                    avg_growth = df['Care Load Growth Rate'].mean()
                    max_growth = df['Care Load Growth Rate'].max()
                    min_growth = df['Care Load Growth Rate'].min()
                    recent_growth = df['Care Load Growth Rate'].iloc[-7:].mean()
                    st.info(f"ℹ️ Average growth rate: {avg_growth:.2f}%.")
                    st.success(f"📈 Maximum growth rate: {max_growth:.2f}%.")
                    st.error(f"📉 Minimum growth rate: {min_growth:.2f}%.")
                    # Three-condition logic
                    if recent_growth <= 0:
                        st.success(f"Stable Load: Average weekly growth is {recent_growth:.2f}% — no immediate surge signal.")
                    elif 0 < recent_growth <= 5:
                        st.warning(f"Rising Load: Average weekly growth is {recent_growth:.2f}% — load increasing. Consider surge planning.")
                    else:  # recent_growth > 5
                        st.error(f"Critical Surge: Average weekly growth is {recent_growth:.2f}% — urgent attention needed to prevent overload.")
                        
            # 4. Backlog Indicator 
            st.markdown("### 📉 Backlog Risk Signals")
            st.info("📊 Backlog Indicator: Sustained Positive Net Intake")     
            fig_backlog = px.line( df, x=df.index, y='Positive Net Intake', title='Daily Sustained Positive Net Intake (Backlog Indicator)',
                labels={
                    'Sustained Positive Net Intake': 'Sustained Positive Net Intake (Binary)',
                    'index': 'Date'
                }
            )
            fig_backlog.update_layout(autosize=True, title=dict(x=0.5, xanchor='center', yanchor="top") )

            # Render with Streamlit using your rule
            st.plotly_chart(fig_backlog, width='stretch')

            with st.expander("ℹ️ More Information & Insights"):
                st.info("ℹ️ This line chart shows the daily sustained positive net intake, "
                        "used as a backlog indicator. A value of 1 indicates periods "
                        "where intake remained positive for consecutive days.")
                st.success("🔍 Identifies periods where net intake remained positive for consecutive days, signaling potential backlog buildup.")
                st.warning("⚠️ Sustained positive net intake may indicate growing operational pressure and potential backlog.")  
                
        except Exception as e:
            st.error(f"❌ Error rendering charts: {e}")
        
        st.subheader("📊 KPI Visualizations")
    
        # Toggle button: True = Surface Plot, False = Trend-line Plot
        show_surface = st.toggle("Show 3D Trend-line/Surface Map", value=True)
    
        try:
            required_cols = ['Net Daily Intake', 'Care Load Growth Rate', 'Total System Load']
            if all(col in df.columns for col in required_cols):
                df_kpi = df.dropna(subset=required_cols).copy()
                df_kpi['Care Load Growth Rate'] = df_kpi['Care Load Growth Rate'].replace([np.inf, -np.inf], np.nan)
                df_kpi.dropna(subset=['Care Load Growth Rate'], inplace=True)
    
                x_vals = df_kpi['Net Daily Intake'].values
                y_vals = df_kpi['Care Load Growth Rate'].values
                z_vals = df_kpi['Total System Load'].values
    
                if show_surface:
                    # 🌐 KPI Surface Map
                    st.markdown("### 🌐 KPI Surface Map")
                    st.caption("Shows smooth interpolated trends of system load across Net Daily Intake and Care Load Growth Rate, helping identify stress zones in the KPI landscape.")
                    
                    # Create grid for interpolation
                    x_finite = x_vals[np.isfinite(x_vals)]
                    y_finite = y_vals[np.isfinite(y_vals)]
                    grid_x, grid_y = np.mgrid[
                        x_finite.min():x_finite.max():100j,
                        y_finite.min():y_finite.max():100j
                    ]
                    grid_z = griddata((x_vals, y_vals), z_vals, (grid_x, grid_y), method='linear')
    
                    fig_viz = go.Figure(data=[
                        go.Surface(
                            z=grid_z,
                            x=grid_x,
                            y=grid_y,
                            colorscale='Viridis',
                            colorbar=dict(title='<b>Total System Load</b>', x=1.0),
                            cmin=z_vals.min(),
                            cmax=z_vals.max()
                        )
                    ])
                    fig_viz.update_layout(
                        title=dict(text='<b>3D Surface Plot: Total System Load vs. Net Intake & Growth Rate</b>', x=0.5, xanchor='center'),
                        scene=dict(
                            xaxis_title='📦 Net Daily Intake',
                            yaxis_title='📈 Care Load Growth Rate (%)',
                            zaxis_title='🧮 Total System Load',
                        ), autosize=True
                    )
    
                else:
                    # 📈 KPI Evolution Path
                    st.markdown("### 📈 KPI Evolution Path")
                    st.caption("Tracks KPI progression over time with color‑coded timestamps, useful for spotting temporal patterns and growth trajectories.")
                    # 3D Trend-line Plot
                    # Convert datetime index to numeric (Unix timestamp)
                    df_kpi['Time_Numeric'] = df_kpi.index.astype(int) / 10**9
    
                    fig_viz = go.Figure(data=[
                        go.Scatter3d(
                            x=df_kpi['Net Daily Intake'],
                            y=df_kpi['Care Load Growth Rate'],
                            z=df_kpi['Total System Load'],
                            mode='lines+markers',
                            marker=dict(
                                size=3,
                                color=df_kpi['Time_Numeric'],
                                colorscale='Plasma',
                                opacity=0.8,
                                colorbar=dict(title='<b>Time (Unix Timestamp)</b>', x=1.0)
                            ),
                            line=dict(color='blue', width=2)
                        )
                    ])
                    fig_viz.update_layout(
                        title=dict(text='<b>3D Line Plot: KPI Evolution Over Time</b>', x=0.5, xanchor='center'),
                        scene=dict(
                            xaxis_title='📦 Net Daily Intake',
                            yaxis_title='📈 Care Load Growth Rate (%)',
                            zaxis_title='🧮 Total System Load'
                        ), autosize=True
                    )
    
                # Render with Streamlit using your rule  
                st.plotly_chart(fig_viz, width='stretch')
    
                # Expander with insights
                with st.expander("ℹ️ More Information & Insights"):
                    if show_surface:
                        st.info("🌐 The surface plot interpolates KPI values to show smooth trends across 📦 Net Daily Intake and 📈 Growth Rate.")
                    else:
                        st.info("📈 The trend‑line plot shows KPI evolution over time, with Plasma colorscale representing ⏳ time progression.")
                    st.success(f"🧮 Average System Load: {z_vals.mean():.2f}")
                    st.warning(f"🚨 Peak System Load: {z_vals.max():.2f}")
            else:
                st.error("Required columns are missing in the dataset for KPI visualization.")
        except Exception as e:
            st.error(f"⚠️ Error generating KPI visualization: {e}")
        
        df['Discharge Offset Ratio'] = df['Children discharged from HHS Care'] / df['Children transferred out of CBP custody'].replace(0, np.nan)
        df['Discharge Offset Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df['Discharge Offset Ratio'].fillna(0, inplace=True)
        
        # 🔄 Dynamic Discharge Cloud
        st.markdown("### 🔄 Dynamic Discharge Cloud")
        st.caption("Visualizes individual data points of intake, discharge ratio, and system load, making it easier to detect anomalies or shifts in discharge effectiveness.")
        try:
            # Ensure necessary columns exist and handle NaNs
            required_cols_discharge_scatter = ['Net Daily Intake', 'Discharge Offset Ratio', 'Total System Load']
            df_discharge_scatter_3d = df.dropna(subset=required_cols_discharge_scatter).copy()
        
            # Convert datetime index to numerical representation for color scaling
            df_discharge_scatter_3d['Time_Numeric'] = df_discharge_scatter_3d.index.astype(int) / 10**9
        
            # Build 3D scatter plot
            fig_discharge_scatter_3d = go.Figure(data=[
                go.Scatter3d( x=df_discharge_scatter_3d['Net Daily Intake'], y=df_discharge_scatter_3d['Discharge Offset Ratio'], 
                            z=df_discharge_scatter_3d['Total System Load'], 
                            mode='markers',
                    marker=dict( size=5, color=df_discharge_scatter_3d['Time_Numeric'], colorscale='Viridis', opacity=0.8, 
                                colorbar=dict(title='⏳ <b>Time (Unix)</b>', x=1.0)
                    ),
                    text=[
                        f"📅 Date: {date.strftime('%Y-%m-%d')}<br>🍽 Net Intake: {ni:.0f}<br>🔄 Discharge Ratio: {dor:.2f}<br>👶 Total Load: {tsl:.0f}"
                        for date, ni, dor, tsl in zip(
                            df_discharge_scatter_3d.index, df_discharge_scatter_3d['Net Daily Intake'], 
                            df_discharge_scatter_3d['Discharge Offset Ratio'],  df_discharge_scatter_3d['Total System Load']
                        )
                    ],
                    hoverinfo='text'
                )
            ])
        
            # Layout with centered bold title
            fig_discharge_scatter_3d.update_layout(
                scene=dict( xaxis_title='🍽 Net Daily Intake', yaxis_title='🔄 Discharge Offset Ratio', zaxis_title='👶 Total System Load' ),
                title=dict( text='📊 <b>3D Scatter Plot: Intake, Discharge Ratio & Load Over Time</b>', x=0.5, xanchor='center', yanchor='top' ),
                autosize=True )
        
            # Render chart in Streamlit
            st.plotly_chart(fig_discharge_scatter_3d, width='stretch')
        
            # Dynamic expander with stats
            with st.expander("ℹ️ More Information & Insights about discharge cloud"):
                total_points = len(df_discharge_scatter_3d)
                avg_intake = df_discharge_scatter_3d['Net Daily Intake'].mean()
                avg_discharge = df_discharge_scatter_3d['Discharge Offset Ratio'].mean()
                avg_load = df_discharge_scatter_3d['Total System Load'].mean()
                latest_date = df_discharge_scatter_3d.index.max()
        
                st.info(f"ℹ️ This scatter plot visualizes {total_points} data points of intake, discharge ratio, and system load over time.")
                st.success(f"✅ Average Net Intake: {avg_intake:.0f}, Average Discharge Ratio: {avg_discharge:.2f}, Average Load: {avg_load:.0f}.")
                st.warning(f"⚠️ Latest data point on 📅 {latest_date.date()} shows 🍽 Intake={df_discharge_scatter_3d.loc[latest_date,'Net Daily Intake']:.0f}, 🔄 Discharge Ratio={df_discharge_scatter_3d.loc[latest_date,'Discharge Offset Ratio']:.2f}, 👶 Load={df_discharge_scatter_3d.loc[latest_date,'Total System Load']:.0f}.")
        
        except Exception as e:
            st.error(f"❌ Error rendering Discharge Scatter 3D chart: {e}")
        
        # 🗺️ Discharge Load Terrain
        st.markdown("### 🗺️ Discharge Load Terrain")
        st.caption("Provides a continuous surface view of how intake and discharge ratios interact to influence system load, highlighting potential bottlenecks or efficiency gaps.")
        try:
            # Ensure necessary columns exist and handle NaNs
            required_cols_discharge_surface = ['Net Daily Intake', 'Discharge Offset Ratio', 'Total System Load']
            df_discharge_surface_3d = df.dropna(subset=required_cols_discharge_surface).copy()
        
            # Prepare data for surface plot
            x_coords = df_discharge_surface_3d['Net Daily Intake'].values
            y_coords = df_discharge_surface_3d['Discharge Offset Ratio'].values
            z_values = df_discharge_surface_3d['Total System Load'].values
        
            # Create grid for interpolation
            x_finite = x_coords[np.isfinite(x_coords)]
            y_finite = y_coords[np.isfinite(y_coords)]
        
            grid_x, grid_y = np.mgrid[ x_finite.min():x_finite.max():100j, y_finite.min():y_finite.max():100j ]
        
            # Interpolate values
            grid_z = griddata((x_coords, y_coords), z_values, (grid_x, grid_y), method='cubic')
        
            # Build 3D surface plot
            fig_discharge_surface_3d = go.Figure(data=[
                go.Surface( z=grid_z, x=grid_x, y=grid_y, colorscale='Plasma', colorbar=dict(title='👶 <b>Total System Load</b>', x=1.0),
                    cmin=z_values.min(), cmax=z_values.max() )
            ])
        
            # Layout with centered bold title
            fig_discharge_surface_3d.update_layout(
                title=dict( text='📊 <b>3D Surface Plot: Load vs. Intake & Discharge Ratio</b>', x=0.5, xanchor='center', yanchor='top' ),
                scene=dict( xaxis_title='🍽 Net Daily Intake', yaxis_title='🔄 Discharge Offset Ratio', zaxis_title='👶 Total System Load' ),
                autosize=True
            )
        
            # Render chart in Streamlit
            st.plotly_chart(fig_discharge_surface_3d, width='stretch')
        
            # Dynamic expander with stats
            with st.expander("ℹ️ More Information & Insights about load terrain"):
                total_points = len(df_discharge_surface_3d)
                avg_intake = df_discharge_surface_3d['Net Daily Intake'].mean()
                avg_discharge = df_discharge_surface_3d['Discharge Offset Ratio'].mean()
                avg_load = df_discharge_surface_3d['Total System Load'].mean()
                busiest_day = df_discharge_surface_3d['Total System Load'].idxmax().date()
        
                st.info(f"ℹ️ This surface plot visualizes {total_points} data points of intake, discharge ratio, and system load.")
                st.success(f"✅ Average Intake: {avg_intake:.0f}, Average Discharge Ratio: {avg_discharge:.2f}, Average Load: {avg_load:.0f}.")
                st.warning(f"⚠️ The highest system load was observed on 📅 {busiest_day}, indicating a potential strain period.")
        
        except Exception as e:
            st.error(f"❌ Error rendering Discharge Surface 3D chart: {e}")
    
    # -------------------------
    # Subtab: Trend & Temporal Analysis
    # -------------------------
    with s_trend:
        st.subheader("Trend & Temporal Analysis")
        st.info("📈 Daily, weekly, monthly trends & 🌡️ Heatmap, 🚦 Threshold analysis, 📊 Time-line comparison, ⚙️ Discharge effectiveness, 🔄 Flow efficiency, ⏱️ Lag analysis, and 🔮 Future predictions preview.")
        st.markdown("### 📈 Daily Total System Load (CBP +/vs HHS)")
        col1, col2 = st.columns(2)
        # 4.0 Daily total system load
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
        
            # Sort by index to ensure chronological order
            df = df.sort_index()
            # Daily Total System Load chart
            fig_daily_CPB_vs_HHS = px.line( df, x=df.index, y='Total System Load', labels={'Total System Load': '👶 Total Children Under Care', 'index': '📅 Date'} )
            fig_daily_CPB_vs_HHS.update_layout(xaxis=dict( title="📅 Date", tickangle=0),
                title=dict(text="<b>📊 CBP + 🏥 HHS Daily Total System Load</b>", x=0.5, xanchor='center', yanchor='top'), height=450 )
        
            with col1:
                st.plotly_chart(fig_daily_CPB_vs_HHS, width='stretch')
                
                with st.expander("ℹ️ More Information & Insights"):
                    avg_load = df['Total System Load'].mean()
                    max_load = df['Total System Load'].max()
                    busiest_day = df['Total System Load'].idxmax().date()
                    latest_row = df.iloc[-1]
                    latest_date = latest_row.name 
                    latest_val = latest_row['Total System Load']
        
                    st.info("ℹ️ This line chart shows the daily total number of children under care across CBP and HHS systems.")
                    st.success(f"✅ Average daily load: {avg_load:,.0f}. Peak load: {max_load:,.0f} on 📅 {busiest_day}.")
                    st.warning(f"⚠️ Latest value on 📅 {latest_date.date()} is {latest_val:,.0f}, which may indicate current operational strain.")
            
            # 4.0.1. CBP vs HHS Load Comparison
            comparison_cols = []
            comparison_cols.append('Children in CBP custody')
            comparison_cols.append('Children in HHS Care')
        
            # Interactive area chart with Plotly
            fig_comparison = px.area( df, x=df.index, y=comparison_cols,
                labels={ 'value': '👶 Number of Children', 'index': '📅 Date', 'variable': '🏢 System'}, title="📊 CBP vs 🏥 HHS Daily Total System Load Comparison")
        
            # Update layout for presentation clarity
            fig_comparison.update_layout( title=dict(x=0.5, xanchor='center'), autosize=True,
                legend=dict(
                    title="🏢 System",
                    orientation="h",       # horizontal legend
                    yanchor="bottom",
                    y=-0.5,                # push legend below chart
                    xanchor="center",
                    x=0.5                  # center the legend
                ),
                xaxis=dict(
                    title="📅 Date",
                    tickangle=0,           # horizontal date labels
                    tickmode="auto",
                    mirror=True
                ), yaxis=dict(title="👶 Children Under Care"))
            with col2:
                # Render with Streamlit using your rule
                st.plotly_chart(fig_comparison, width='stretch')
                
                with st.expander("ℹ️ More Information & Insights"):
                    if 'Children in CBP custody' in df.columns and 'Children in HHS Care' in df.columns:
                        corr = df['Children in CBP custody'].corr(df['Children in HHS Care'])
                        st.metric("CBP vs HHS correlation", f"{corr:.2f}")
                    for col in comparison_cols:
                        avg_val = df[col].mean()
                        max_val = df[col].max()
                        busiest_day = df[col].idxmax().date()
                        st.info(f"ℹ️ {col}: Avg = {avg_val:,.0f}, Peak = {max_val:,.0f} on 📅 {busiest_day}")
        
        except Exception as e:
            st.error(f"❌ Error rendering Daily Total System Load chart: {e}")
        
        # 4.0.1 Weekly and monthly aggregated trends
        st.markdown("### 📊 Weekly & Monthly Aggregated Trends")
        try:
            weekly = df['Total System Load'].resample('W').sum()
            monthly = df['Total System Load'].resample('ME').sum()
            fig_trends = go.Figure()
            fig_trends.add_trace(go.Scatter(x=weekly.index, y=weekly.values, mode='lines', name='Weekly Load'))
            fig_trends.add_trace(go.Scatter(x=monthly.index, y=monthly.values, mode='lines', name='Monthly Load'))
            fig_trends.update_layout( title=dict( text="📊 <b>Weekly and Monthly Total System Load</b>", x=0.5, xanchor='center', yanchor='top', 
                automargin=True, font=dict(size=18) ), 
                legend=dict(title="📌 Trends", orientation="h", yanchor="bottom", y=-0.4, x=0.5, xanchor="center"), autosize=True)
            fig_trends.update_xaxes(title_text="📅 Date")
            fig_trends.update_yaxes(title_text="⚡ Total System Load")
            
            st.plotly_chart(fig_trends, width='stretch')
            # Dynamic expander with stats
            with st.expander("ℹ️ More Information & Insights"):
                avg_weekly = weekly.mean()
                avg_monthly = monthly.mean()
                peak_week = weekly.idxmax().date()
                peak_month = monthly.idxmax().strftime("%B %Y")
        
                st.info("ℹ️ This chart compares weekly and monthly aggregated system load trends.")
                st.success(f"✅ Average weekly load: {avg_weekly:,.0f}, average monthly load: {avg_monthly:,.0f}.")
                st.warning(f"⚠️ Peak weekly load occurred on 📅 {peak_week}. Peak monthly load was in {peak_month}.")
        
        except Exception as e:
            st.error(f"❌ Error rendering Weekly/Monthly trends chart: {e}")
    
        # 4.0.2 High-Load Thresholds
        st.markdown("### ⚠️ High‑Load Threshold Analysis (90th Percentile)")
        try:
            # Identification of sustained high-load periods
            threshold = df['Total System Load'].quantile(0.90)  # Top 10% as high load
            high_load_periods = df[df['Total System Load'] > threshold]
        
            fig_highload = px.line( df, x=df.index, y='Total System Load', labels={'Total System Load': '👶 Total Children Under Care', 'index': '📅 Date'} )
        
            # Add threshold line
            fig_highload.add_hline( y=threshold, line_dash="dash", line_color="red", annotation_text="🔴 90th Percentile Threshold" )
        
            # Highlight high-load points
            fig_highload.add_trace(
                go.Scatter( x=high_load_periods.index, y=high_load_periods['Total System Load'], mode='markers', 
                        marker=dict(color='red', size=8, symbol="diamond"), name='High Load Periods 🔥' )
            )
        
            # Update layout with centered bold title
            fig_highload.update_layout(title=dict( text=f'⚠️ <b>Total System Load with High-Load Threshold ({threshold:.0f})</b>',
                    x=0.5, xanchor='center', yanchor='top', automargin=True
                ), legend=dict(title="📌 Metric", orientation="h", yanchor="bottom", y=-0.4, x=0.5, xanchor="center"), autosize=True
            )
        
            # Render chart in Streamlit with unique key
            st.plotly_chart(fig_highload, width='stretch')
        
            # Dynamic expander content based on current filtered data
            with st.expander("ℹ️ More Information & Insights"):
                total_days = len(df)
                high_days = len(high_load_periods)
                high_pct = (high_days / total_days * 100) if total_days > 0 else 0
                latest_high = high_load_periods.index.max() if not high_load_periods.empty else None
        
                st.info(f"ℹ️ Out of 📅 {total_days} days in the selected range, 🔥 {high_days} days ({high_pct:.1f}%) exceeded the high-load threshold.")
                if latest_high:
                    st.success(f"✅ The most recent high-load period occurred on 📅 {latest_high.date()}, with 👶 {int(high_load_periods.loc[latest_high, 'Total System Load'])} children under care.")
                else:
                    st.success("✅ No high-load periods detected in the current selection.")
                st.warning("⚠️ Sustained high-load periods may indicate operational strain, requiring proactive resource allocation.")
        
        except Exception as e:
            st.error(f"❌ Error rendering High-Load Periods chart: {e}")
            
        st.markdown("### 📊 Timeline Comparison: Total System Load")
        try:
            # --- Month options ---
            month_options = pd.date_range(df.index.min(), df.index.max(), freq="MS").strftime("%Y-%m").tolist()
            total_months = len(month_options)
        
            st.markdown("### Select Custom Timelines")
        
            if total_months >= 3:
                # --- Divide into thirds using months ---
                one_third = total_months // 3
                two_third = 2 * total_months // 3
        
                early_range_default = month_options[:one_third]       # first 1/3
                late_range_default  = month_options[two_third:]       # last 1/3
        
                colA, colB = st.columns(2)
        
                with colA:
                    early_range = st.select_slider(
                        "Select Early Period (Months)",
                        options=month_options,
                        value=(early_range_default[0], early_range_default[-1])
                    )
                    early_period_start = pd.to_datetime(early_range[0])
                    early_period_end   = pd.to_datetime(early_range[1]) + pd.offsets.MonthEnd(0)
        
                with colB:
                    late_range = st.select_slider(
                        "Select Late Period (Months)",
                        options=month_options,
                        value=(late_range_default[0], late_range_default[-1])
                    )
                    late_period_start = pd.to_datetime(late_range[0])
                    late_period_end   = pd.to_datetime(late_range[1]) + pd.offsets.MonthEnd(0)
        
            else:
                # --- Fallback to daily ranges with slider ---
                st.warning("⚠️ Not enough months to slice so falling back to daily ranges.")
        
                day_options = pd.date_range(df.index.min(), df.index.max(), freq="D").strftime("%Y-%m-%d").tolist()
                total_days = len(day_options)
        
                one_third = max(1, total_days // 3)
                two_third = 2 * total_days // 3
        
                early_range_default = day_options[:one_third]
                late_range_default  = day_options[two_third:]
                # Ensure at least 2 days in defaults
                if len(early_range_default) == 1 and total_days > 1:
                    early_range_default.append(day_options[1])
                if len(late_range_default) == 1 and total_days > 1:
                    late_range_default.insert(0, day_options[-2])
        
                colA, colB = st.columns(2)
        
                with colA:
                    early_range = st.select_slider(
                        "Select Early Period (Days)",
                        options=day_options,
                        value=(early_range_default[0], early_range_default[-1])
                    )
                    early_period_start = pd.to_datetime(early_range[0])
                    early_period_end   = pd.to_datetime(early_range[1])
        
                with colB:
                    late_range = st.select_slider(
                        "Select Late Period (Days)",
                        options=day_options,
                        value=(late_range_default[0], late_range_default[-1])
                    )
                    late_period_start = pd.to_datetime(late_range[0])
                    late_period_end   = pd.to_datetime(late_range[1])
        
            # --- Filter data ---
            df_early = df[(df.index >= early_period_start) & (df.index <= early_period_end)]
            df_late  = df[(df.index >= late_period_start) & (df.index <= late_period_end)]
        
            # --- Create subplots ---
            fig_timeline_comparison = make_subplots(
                rows=2, cols=1,
                subplot_titles=[
                    f'Early Timeline ({early_period_start.date()} to {early_period_end.date()})',
                    f'Late Timeline ({late_period_start.date()} to {late_period_end.date()})'
                ], shared_xaxes=False, vertical_spacing=0.4
            )
        
            if not df_early.empty:
                fig_timeline_comparison.add_trace(
                    go.Scatter(x=df_early.index, y=df_early['Total System Load'], mode='lines', name='Early Period'), row=1, col=1
                )
        
            if not df_late.empty:
                fig_timeline_comparison.add_trace(
                    go.Scatter(x=df_late.index, y=df_late['Total System Load'], mode='lines', name='Late Period', line=dict(color='orange')), row=2, col=1
                )
        
            fig_timeline_comparison.update_layout(
                title_text="<b>Comparison Across Custom Periods: Total System Load</b>",
                autosize=True,
                margin=dict(t=80, b=80),
                legend=dict(title="📌 Scale", orientation="h", yanchor="bottom", y=-0.4, x=0.5, xanchor="center"),
                title=dict(x=0.5, xanchor='center', font=dict(size=17))
            )
        
            st.plotly_chart(fig_timeline_comparison, width='stretch')
        
            with st.expander("ℹ️ More Information & Insights"):
                st.info("📊 This chart compares the total system load across the first and last thirds of the timeline. 🎚️ Adjust the sliders above to explore different ranges.")
                if not df_early.empty:
                    st.success(f"🟢 Early Period Avg Load: {df_early['Total System Load'].mean():,.0f}")
                if not df_late.empty:
                    st.success(f"🔵 Late Period Avg Load: {df_late['Total System Load'].mean():,.0f}")
        
        except Exception as e:
            st.error(f"❌ Error rendering timeline comparison chart: {e}")
        
        # 4.1 Monthly heatmap
        st.markdown("### 📊 Monthly Total System Load Heatmap")
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
        
            # Prepare monthly load data
            monthly_load_heatmap_data = df['Total System Load'].resample('ME').sum().to_frame()
            monthly_load_heatmap_data['Year'] = monthly_load_heatmap_data.index.year
            monthly_load_heatmap_data['Month'] = monthly_load_heatmap_data.index.month_name()
            
            # Pivot for heatmap
            month_order = [
                'January', 'February', 'March', 'April', 'May', 'June',
                'July', 'August', 'September', 'October', 'November', 'December'
            ]
            monthly_load_pivot = monthly_load_heatmap_data.pivot(
                index='Year', columns='Month', values='Total System Load'
            )
            
            # Only keep months that exist in the data
            available_months = [m for m in month_order if m in monthly_load_pivot.columns]
            monthly_load_pivot = monthly_load_pivot[available_months]
            
            # Plotly heatmap
            fig_monthly_heatmap = go.Figure(data=go.Heatmap(
                z=monthly_load_pivot.values,
                x=monthly_load_pivot.columns,
                y=monthly_load_pivot.index,
                colorscale='Viridis',
                text=monthly_load_pivot.values,
                texttemplate="%{text:,.0f}",
                hovertemplate="Year %{y}, %{x}: %{z:,.0f}<extra></extra>"
            ))
            
            fig_monthly_heatmap.update_layout(
                title=dict( text="📊 <b>Monthly Total System Load Heatmap</b>", x=0.5, xanchor='center', yanchor='top', font=dict(size=18) ),
                xaxis_title="📅 Month", yaxis_title="📆 Year", autosize=True
            )
        
            # Render chart in Streamlit
            st.plotly_chart(fig_monthly_heatmap, width='stretch')
        
            # Dynamic expander with stats
            with st.expander("ℹ️ More Information & Insights"):
                total_years = monthly_load_pivot.index.nunique()
                avg_load = monthly_load_heatmap_data['Total System Load'].mean()
                max_month = monthly_load_heatmap_data.loc[
                    monthly_load_heatmap_data['Total System Load'].idxmax()
                ]
                busiest_month = max_month['Month']
                busiest_year = max_month['Year']
        
                st.info(f"ℹ️ This heatmap shows monthly aggregated system load across {total_years} years.")
                st.success(f"✅ Average monthly load: {avg_load:,.0f}.")
                st.warning(f"⚠️ The busiest period was {busiest_month} {busiest_year}, with the highest load observed.")
        
        except Exception as e:
            st.error(f"❌ Error rendering Monthly Load Heatmap: {e}")            
        
        # 4.2 Discharge Effectiveness Over Time
        st.markdown("### ⚖️ Discharge Effectiveness Over Time")
        
        avg_offset = df['Discharge Offset Ratio'].mean()
        
        fig_offset = px.line(df, x=df.index, y='Discharge Offset Ratio')
        fig_offset.add_hline(y=avg_offset, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_offset:.2f}")
        fig_offset.update_layout( 
        title=dict(text="⚖️ <b>Discharge Offset Ratio Over Time</b>", x=0.5, xanchor='center', yanchor='top'), autosize=True)
        st.plotly_chart(fig_offset, width='stretch')
        
        with st.expander("ℹ️ More Information"):
            latest_date = df.index.max()
            latest_val = df.loc[latest_date, 'Discharge Offset Ratio']
            st.info("ℹ️ Ratio of discharges to transfers — measures outflow effectiveness relative to inflow.")
            st.success(f"✅ Average ratio: {avg_offset:.2f}. Latest value on 📅 {latest_date.date()} is {latest_val:.2f}.")
            st.warning("⚠️ Low values suggest backlog risk, while high values indicate strong discharge throughput.")
            # ⚖️ Discharge Effectiveness
            if avg_offset < 1.0:
                st.warning(f"🐢 Discharge Lag: Ratio = {avg_offset:.2f} — discharges lag transfers ⚠️ backlog risk.")
            elif avg_offset == 1.0:
                st.info(f"⚖️ Discharge Balanced: Ratio = {avg_offset:.2f} — outflow exactly matches inflow 📊 steady state.")
            else:  # avg_offset > 1.0
                st.success(f"🚀 Discharge Healthy: Ratio = {avg_offset:.2f} — discharges exceed transfers ✅ system catching up.")
            
        # 4.3 Lag Analysis & Feature Statistics
        st.markdown("### 🔄 Lag Analysis & Feature Statistics")
        df_lag = df[['Total System Load']].copy()
        for lag in range(1, 15):
            df_lag[f'lag_{lag}'] = df_lag['Total System Load'].shift(lag)
        df_lag.dropna(inplace=True)
        
        corr = df_lag.corr()
        fig_corr = px.imshow(corr, color_continuous_scale='RdBu', zmin=-1, zmax=1)
        fig_corr.update_layout(autosize=True,
        title=dict(text="📈 <b>Lag Correlation Matrix (Total System Load & lags)</b>", x=0.5, xanchor='center', yanchor='top', font=dict(size=16)))
        st.plotly_chart(fig_corr, width='stretch')
        
        with st.expander("ℹ️ More Information & Insights"):
            st.info("ℹ️ Lag correlations help identify persistence and autoregressive structure.")
            st.success("✅ High correlation at short lags suggests strong inertia in load.")
        
        # 4.4 Flow Efficiency Scatter
        st.markdown("### 🔀 Flow Efficiency: Net Intake vs Discharge Offset Ratio")
        
        df_flow = df.dropna(subset=['Net Daily Intake', 'Discharge Offset Ratio'])
        if not df_flow.empty:
            fig_flow = px.scatter(
                df_flow,
                x='Net Daily Intake',
                y='Discharge Offset Ratio',
                marginal_x='histogram',
                marginal_y='histogram',
                trendline='ols',
                labels={'Net Daily Intake': '📦 Net Daily Intake', 'Discharge Offset Ratio': '⚖️ Discharge Offset Ratio'}
            )
            fig_flow.update_layout(autosize=True,
            title=dict( text="🔀 <b>Net Intake vs Discharge Offset Ratio</b>", x=0.5, xanchor='center', yanchor='top', font=dict(size=16))
            )
            st.plotly_chart(fig_flow, width='stretch', key="flow_efficiency_chart")
        
            with st.expander("ℹ️ More Information & Insights"):
                avg_intake = df_flow['Net Daily Intake'].mean()
                avg_discharge = df_flow['Discharge Offset Ratio'].mean()
        
                st.info("📊 Joint distribution with marginals helps identify dense operational states.")
                st.success(f"📦 Average Net Intake: {avg_intake:.0f}, ⚖️ Average Discharge Ratio: {avg_discharge:.2f}")
        
                # Dynamic condition presentation with symbols
                if avg_discharge < 1.0:
                    st.warning(f"🐢 Discharge Lag: Avg ratio {avg_discharge:.2f} — inflow exceeds outflow ⚠️ backlog risk.")
                elif avg_discharge == 1.0:
                    st.info(f"⚖️ Balanced Flow: Avg ratio {avg_discharge:.2f} — inflow matches outflow 📊 steady state.")
                else:
                    st.success(f"🚀 Healthy Flow: Avg ratio {avg_discharge:.2f} — discharges exceed transfers ✅ system catching up.")
        
        else:
            st.info("ℹ️ Insufficient data for flow efficiency scatter.")
        
        # 4.5.0 3D Trend Line
        try:
            st.markdown("### 📈 3D Trend Line: Load, Rolling Avg & Net Intake")
            st.info("📝 A combined view of load, rolling averages, and intake, plotted in three dimensions to reveal underlying trends and relationships")
            # Compute rolling average
            df['7-Day Rolling Avg Load'] = df['Total System Load'].rolling(window=7).mean()
        
            # Ensure necessary columns exist and handle NaNs
            required_cols_trend_line_3d = ['Total System Load', '7-Day Rolling Avg Load', 'Net Daily Intake']
            df_trend_line_3d = df.dropna(subset=required_cols_trend_line_3d).copy()
        
            # Convert datetime index to numerical representation for color scaling
            df_trend_line_3d['Time_Numeric'] = df_trend_line_3d.index.astype(int) / 10**9
        
            # Build 3D line plot
            fig_trend_line_3d = go.Figure(data=[
                go.Scatter3d(
                    x=df_trend_line_3d['Total System Load'],
                    y=df_trend_line_3d['7-Day Rolling Avg Load'],
                    z=df_trend_line_3d['Net Daily Intake'],
                    mode='lines+markers',
                    marker=dict(
                        size=3,
                        color=df_trend_line_3d['Time_Numeric'],
                        colorscale='Viridis',
                        opacity=0.8,
                        colorbar=dict(title='⏳ Time (Unix)', x=1.0)
                    ),
                    line=dict(color='purple', width=2)
                )
            ])
        
            # Layout with centered bold title
            fig_trend_line_3d.update_layout(
                scene=dict( xaxis_title='⚙️ Total System Load', yaxis_title='📊 7-Day Rolling Avg Load', zaxis_title='🍽 Net Daily Intake' ),
                title=dict(text='📈 <b>3D Line Plot: Load, Rolling Avg & Net Intake</b>', x=0.5, xanchor='center', yanchor='top'), 
                autosize=True
            )
        
            # Render chart in Streamlit
            st.plotly_chart(fig_trend_line_3d, width='stretch', key="trend_line_3d_chart")
        
            # Dynamic expander with stats
            with st.expander("ℹ️ More Information & Insights"):
                total_days = len(df_trend_line_3d)
                avg_load = df_trend_line_3d['Total System Load'].mean()
                avg_rolling = df_trend_line_3d['7-Day Rolling Avg Load'].mean()
                avg_intake = df_trend_line_3d['Net Daily Intake'].mean()
                latest_date = df_trend_line_3d.index.max() if not df_trend_line_3d.empty else None
        
                st.info(f"ℹ️ This plot shows {total_days} days of system data, combining load, rolling average, and net intake.")
                st.success(f"✅ Average Load: {avg_load:.0f}, Rolling Avg: {avg_rolling:.0f}, Net Intake: {avg_intake:.0f}")
                if latest_date:
                    st.warning(f"⚠️ Latest data point on 📅 {latest_date.date()} shows 👶 Load={df_trend_line_3d.loc[latest_date,'Total System Load']:.0f}, Rolling={df_trend_line_3d.loc[latest_date,'7-Day Rolling Avg Load']:.0f}, Intake={df_trend_line_3d.loc[latest_date,'Net Daily Intake']:.0f}")
        
        except Exception as e:
            st.error(f"❌ Error rendering 3D Trend Line chart: {e}")
        
        # 4.5.1 Temporal Surface 3D
        st.markdown("### 📅 3D Temporal Surface: Load vs Day & Month")
        try:
            st.info("📝 A combined weekday–month view of system load, showing peaks and troughs across time dimensions for proactive planning")
            # Ensure 'Day_of_Week' and 'Month' features exist
            if 'Day_of_Week' not in df.columns:
                df['Day_of_Week'] = df.index.dayofweek
            if 'Month' not in df.columns:
                df['Month'] = df.index.month
        
            # Ensure necessary columns exist and handle NaNs
            required_cols_temporal_surface_3d = ['Total System Load', 'Day_of_Week', 'Month']
            df_temporal_surface_3d = df.dropna(subset=required_cols_temporal_surface_3d).copy()
        
            # Prepare data for surface plot
            x_coords = df_temporal_surface_3d['Day_of_Week'].values
            y_coords = df_temporal_surface_3d['Month'].values
            z_values = df_temporal_surface_3d['Total System Load'].values
        
            # Create grid for interpolation
            x_finite = x_coords[np.isfinite(x_coords)]
            y_finite = y_coords[np.isfinite(y_coords)]
            grid_x, grid_y = np.mgrid[ x_finite.min():x_finite.max():10j, y_finite.min():y_finite.max():12j ]
        
            # Interpolate values
            grid_z = griddata( (x_coords, y_coords), z_values, (grid_x, grid_y), method='linear' )
        
            # Build 3D surface plot
            fig_temporal_surface_3d = go.Figure(data=[
                go.Surface( z=grid_z, x=grid_x, y=grid_y, colorscale='Plasma', colorbar=dict(title='👶 <b>Total System Load</b>', x=1.0), 
                        cmin=z_values.min(), cmax=z_values.max()
                )
            ])
        
            # Layout with centered bold title
            fig_temporal_surface_3d.update_layout(
                title=dict(
                    text='📅 <b>3D Surface Plot: Load vs. Day of Week & Month</b>', x=0.5, xanchor='center', yanchor='top'
                ),
                margin=dict(l=0, r=0, b=50, t=50),
                scene=dict( xaxis_title='📆 Day of Week (0=Mon, 6=Sun)', yaxis_title='🗓 Month (1=Jan, 12=Dec)', zaxis_title='⚙️ Total System Load'),
                autosize=True
            )
        
            # Render chart in Streamlit
            st.plotly_chart(fig_temporal_surface_3d, width='stretch')
        
            # Dynamic expander with stats
            with st.expander("ℹ️ More Information & Insights"):
                total_days = len(df_temporal_surface_3d)
                avg_load = df_temporal_surface_3d['Total System Load'].mean()
                busiest_day = int(df_temporal_surface_3d.groupby('Day_of_Week')['Total System Load'].mean().idxmax())
                busiest_month = int(df_temporal_surface_3d.groupby('Month')['Total System Load'].mean().idxmax())
        
                st.info(f"ℹ️ This surface plot covers {total_days} days of data, showing temporal load patterns across weekdays and months.")
                st.success(f"✅ Average Load: {avg_load:.0f}. The busiest weekday is 📆 {busiest_day} (0=Mon), and the busiest month is 🗓 {busiest_month}.")
                st.warning("⚠️ Peaks in certain months or weekdays may indicate recurring demand cycles that require proactive planning.")
        
        except Exception as e:
            st.error(f"❌ Error rendering Temporal Surface 3D chart: {e}")
        
    # -------------------------
    # Subtab: Pressure & Stress (separate as requested)
    # -------------------------
    with s_pressure:
        st.subheader("Pressure & Stress Systems")
        st.info("📈 Rolling Averages, 🟥 Strain Windows, 🌐 3D Load–Variability Maps, and 🌡️ Stress/Pressure Heatmaps.")
        st.markdown("### 🔄 Daily Load With Rolling Trends")

        # Default rolling averages
        df['7-Day Rolling Avg Load'] = df['Total System Load'].rolling(window=7).mean()
        df['14-Day Rolling Avg Load'] = df['Total System Load'].rolling(window=14).mean()
        
        # Horizontal scroll bar (slider) for optional custom window
        with st.expander("🎚️ Compare Your Custom Rolling Average Window"):
            custom_window = st.slider(
                "📊 Optional rolling window (days)",
                min_value=0, max_value=60, value=0, step=1,
                help="Select 0 for no custom window, or choose any value between 2–60 days other than 7 or 14 to add a custom rolling average line to the chart."
            )
            # Define excluded values
            excluded_windows = [0, 1, 7, 14]
            if custom_window in excluded_windows:
                st.info("ℹ️ Current Selection: No custom rolling average selected")
            else:
                st.success(f"✅ Current Selection: {custom_window}-day rolling average")
        
        # Check for invalid selections
        if custom_window in [1, 7, 14]:
            st.warning(f"⚠️ The {custom_window}-day window is already included by default. Please choose another value.")
            custom_window = 0  # reset to no custom window
        
        # Add custom rolling average if chosen and not equal to 0, 7, or 14
        if custom_window not in [0, 1, 7, 14]:
            df[f'{custom_window}-Day Rolling Avg Load'] = df['Total System Load'].rolling(window=custom_window).mean()
        
        try:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
        
            # Interactive line chart with rolling averages
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df.index, y=df['Total System Load'],
                mode='lines', name='📅 Daily Load',
                line=dict(color='blue'), opacity=0.6
            ))
            fig1.add_trace(go.Scatter(
                x=df.index, y=df['7-Day Rolling Avg Load'],
                mode='lines', name='📊 7-Day Rolling Avg',
                line=dict(color='orange')
            ))
            fig1.add_trace(go.Scatter(
                x=df.index, y=df['14-Day Rolling Avg Load'],
                mode='lines', name='📊 14-Day Rolling Avg',
                line=dict(color='green')
            ))
        
            # Add custom rolling average line if selected
            if custom_window not in [0, 1, 7, 14]:
                fig1.add_trace(go.Scatter(
                    x=df.index, y=df[f'{custom_window}-Day Rolling Avg Load'],
                    mode='lines', name=f'📊 {custom_window}-Day Rolling Avg',
                    line=dict(color='yellow')
                ))
        
            # Layout with centered bold title
            fig1.update_layout(
                title=dict( text="📈 <b>Total System Load with Rolling Averages</b>", x=0.5, xanchor='center', yanchor='top'),
                legend=dict(title="📌 Rolling Trends", orientation="h", yanchor="bottom", y=-0.66, x=0.5, xanchor="center"), 
                xaxis_title="📅 Date", yaxis_title="👶 Total Children Under Care", autosize=True
            )
        
            # Render chart in Streamlit
            st.plotly_chart(fig1, width='stretch')
        
            # Dynamic expander with stats       
            with st.expander("ℹ️ More Information & Insights"):
                if df.empty:
                    st.info("ℹ️ No data available to compute statistics.")
                else:
                    # Case 1: No custom window selected
                    if custom_window == 0:
                        st.info("ℹ️ This chart shows daily system load with the default 7‑day and 14‑day rolling averages.")
                    # Case 2: Invalid custom window (1, 7, 14) already handled by warning popup
                    elif custom_window in [1, 7, 14]:
                        st.info("ℹ️ You selected a window that is already included by default. No extra line was added.")
                    else:
                        st.info(f"ℹ️ This chart shows daily system load with 7‑day, 14‑day, and an additional {custom_window}-day rolling average.")
            
                    avg_daily = df['Total System Load'].mean()
                    avg_7day = df['7-Day Rolling Avg Load'].mean()
                    avg_14day = df['14-Day Rolling Avg Load'].mean()
            
                    latest_date = df.index[-1]
                    latest_daily = df['Total System Load'].iloc[-1]
                    latest_7day = df['7-Day Rolling Avg Load'].iloc[-1]
                    latest_14day = df['14-Day Rolling Avg Load'].iloc[-1]
            
                    # Build table rows
                    data = {
                        "Window": ["➡️ Daily", "➡️ 7-Day Avg", "➡️ 14-Day Avg"],
                        "Average Load": [f"{avg_daily:,.0f}", f"{avg_7day:,.0f}", f"{avg_14day:,.0f}"],
                        f"Latest ({latest_date.date()})": [f"{latest_daily:,.0f}", f"{latest_7day:,.0f}", f"{latest_14day:,.0f}"]
                    }
            
                    # Add custom window if valid
                    if custom_window not in [0, 1, 7, 14]:
                        avg_custom = df[f'{custom_window}-Day Rolling Avg Load'].mean()
                        latest_custom = df[f'{custom_window}-Day Rolling Avg Load'].iloc[-1]
                        data["Window"].append(f"➡️ {custom_window}-Day Avg")
                        data["Average Load"].append(f"{avg_custom:,.0f}")
                        data[f"Latest ({latest_date.date()})"].append(f"{latest_custom:,.0f}")
            
                    # Display table
                    st.table(pd.DataFrame(data))
                    st.success("ℹ️ This table summarizes average and latest values for each rolling window.")
            
        except Exception as e:
            st.error(f"❌ Error rendering Rolling Averages chart: {e}")
        
        # Detect prolonged strain windows and highlight on Total System Load graph
        df['Sustained Positive Net Intake'] = (df['Net Daily Intake'] > 0).rolling(window=3).apply(lambda x: x.all(), raw=True).fillna(0)
        mean_load = df['Total System Load'].mean()
        df['High Load'] = (df['Total System Load'] > mean_load).astype(int)
        strain = df[(df['Sustained Positive Net Intake'] == 1) & (df['High Load'] == 1)].copy()

        # Detection of prolonged strain windows
        st.markdown("### ⚡ Prolonged Strain Windows on Total System Load")
        df['Sustained Positive Net Intake'] = ( (df['Net Daily Intake'] > 0) .rolling(window=3) .apply(lambda x: x.all(), raw=True) .fillna(0) )
        mean_load = df['Total System Load'].mean()
        df['High Load'] = (df['Total System Load'] > mean_load).astype(int)
        strain_windows = df[(df['Sustained Positive Net Intake'] == 1) & (df['High Load'] == 1)].copy()
        
        try:
            if not strain_windows.empty:
                # Group consecutive strain days
                strain_windows['group'] = (strain_windows.index.to_series().diff().dt.days > 1).cumsum()
        
                # Build summary table
                summary_data = []
                for _, group_df in strain_windows.groupby('group'):
                    summary_data.append({
                        "➡️ Window Start": group_df.index.min().strftime('%Y-%m-%d'), "➡️ Window End": group_df.index.max().strftime('%Y-%m-%d'),
                        "📊 Days": len(group_df), "👶 Avg Load": f"{group_df['Total System Load'].mean():,.0f}"
                    })
                summary_df = pd.DataFrame(summary_data)
        
                # Chart with highlighted strain windows
                fig_strain_windows = go.Figure()
                fig_strain_windows.add_trace(go.Scatter( x=df.index, y=df['Total System Load'], mode='lines', name='📅 Daily Load', line=dict(color='blue')))
                fig_strain_windows.add_trace(go.Scatter( x=strain_windows.index, y=strain_windows['Total System Load'], mode='markers', name='⚠️ Strain Window',
                                        marker=dict(color='red', size=8)))
        
                fig_strain_windows.update_layout(
                    title=dict( text="📈 <b>Detected Strain Windows on Total System Load</b>", x=0.5, xanchor='center'),
                    legend=dict(title="📌 Scale", orientation="h", yanchor="bottom", y=-0.4, x=0.5, xanchor="center"), 
                    xaxis_title="📅 Date", yaxis_title="👶 Total Children Under Care", autosize=True
                )
        
                # Render chart
                st.plotly_chart(fig_strain_windows, width='stretch')
        
                # Dynamic expander with tabular summary
                with st.expander("ℹ️ More Information & Insights about detected strain windows"):
                    st.info("ℹ️ These windows represent periods of sustained positive net intake (≥3 days) combined with above‑average system load.")
                    st.table(summary_df)
        
            else:
                st.info("ℹ️ No prolonged strain windows detected under the defined conditions.")
        
        except Exception as e:
            st.error(f"❌ Error rendering strain windows chart: {e}")
        
        try:
            st.markdown("### 🌐 3D Load–Variability With Intake & Strain Maps")
            # Ensure dataframe index is named 'Date'
            if df.index.name != 'Date':
                df.index.name = 'Date'
        
            # Calculate rolling std dev
            df['7-Day Rolling Std Dev Load'] = df['Total System Load'].rolling(window=7).std()
        
            # --- First 3D Scatter Plot: Load, Variability, Net Intake ---
            fig_load_variability = go.Figure(data=[
                go.Scatter3d( x=df['Total System Load'], y=df['7-Day Rolling Std Dev Load'], z=df['Net Daily Intake'], mode='markers',
                    marker=dict( size=5, color=df['Total System Load'], colorscale='Viridis', opacity=0.8, 
                                colorbar=dict(title='👶 <b>Total System Load</b>', x=1.0))
                )
            ])
        
            fig_load_variability.update_layout(
                scene=dict( xaxis_title='⚙️ Total System Load', yaxis_title='📊 7-Day Rolling Std Dev Load', zaxis_title='🍽 Net Daily Intake' ),
                title=dict( text='📈 <b>3D Scatter Plot: Load, Variability & Net Intake</b>', x=0.5, xanchor='center', yanchor='top' ),
                autosize=True
            )
        
            st.plotly_chart(fig_load_variability, width='stretch')
        
            with st.expander("ℹ️ More Information & Insights"):
                avg_std = df['7-Day Rolling Std Dev Load'].mean()
                avg_intake = df['Net Daily Intake'].mean()
                latest_date = df.index.max()
                latest_val = df.loc[latest_date, 'Net Daily Intake']
                st.info(f"ℹ️ This scatter plot shows {len(df)} data points combining load, variability, and intake.")
                st.success(f"✅ Average variability: {avg_std:.2f}, average intake: {avg_intake:.2f}. Latest intake on 📅 {latest_date.date()} is {latest_val:.2f}.")
                st.warning("⚠️ High variability may indicate unstable intake patterns requiring closer monitoring.")
        
            # --- Second 3D Scatter Plot: Load, Variability, Strain Windows ---
            fig_2 = go.Figure(data=[
                go.Scatter3d(
                    x=df['Total System Load'],
                    y=df['7-Day Rolling Std Dev Load'],
                    z=df['High Load'],
                    mode='markers',
                    marker=dict( size=5, color=df['Sustained Positive Net Intake'], colorscale='Plasma', opacity=0.8,
                        colorbar=dict(title='🔥 <b>Sustained Positive Net Intake</b>', x=1.0))
                )
            ])
        
            fig_2.update_layout(
                scene=dict(xaxis_title='⚙️ Total System Load', yaxis_title='📊 7-Day Rolling Std Dev Load', zaxis_title='🚨 High Load (0=No, 1=Yes)'),
                title=dict(text='📉 <b>3D Scatter Plot: Load, Variability & Strain Windows</b>', x=0.5, xanchor='center', yanchor='top'),
                autosize=True
            )
        
            st.plotly_chart(fig_2, width='stretch')
        
            with st.expander("ℹ️ More Information & Insights"):
                high_load_days = int(df['High Load'].sum())
                sustained_days = int(df['Sustained Positive Net Intake'].sum())
                st.info(f"ℹ️ This scatter plot highlights strain windows where intake and load are both elevated.")
                st.success(f"✅ High load detected on {high_load_days} days. Sustained positive net intake observed on {sustained_days} days.")
                st.warning("⚠️ Overlap of high load and sustained intake signals prolonged strain periods that may need intervention.")
        
        except Exception as e:
            st.error(f"❌ Error rendering 3D scatter plots: {e}")

        
        st.markdown("### 💢 Stress-Pressure Dynamics")
        try:
            # Ensure required columns exist and handle NaNs
            required_cols = ['Total System Load', '7-Day Rolling Std Dev Load', 'Net Daily Intake', 'Sustained Positive Net Intake', 'High Load']
            df_plot = df.dropna(subset=required_cols).copy()
        
            # Composite Stress Score calculation
            df_plot['Stress_Score'] = df_plot['Total System Load'] * df_plot['7-Day Rolling Std Dev Load']
            df_plot['Stress_Score'] *= (1 + df_plot['Sustained Positive Net Intake'] + df_plot['High Load'])
        
            min_net_intake = df_plot['Net Daily Intake'].min()
            max_net_intake = df_plot['Net Daily Intake'].max()
            if (max_net_intake - min_net_intake) == 0:
                df_plot['Normalized_Net_Intake'] = 0.5
            else:
                df_plot['Normalized_Net_Intake'] = (df_plot['Net Daily Intake'] - min_net_intake) / (max_net_intake - min_net_intake)
        
            df_plot['Stress_Score'] *= (1 + df_plot['Normalized_Net_Intake'])
        
            # Prepare data for surface plot
            x_coords = df_plot['Total System Load'].values
            y_coords = df_plot['7-Day Rolling Std Dev Load'].values
            z_values = df_plot['Stress_Score'].values
        
            grid_x, grid_y = np.mgrid[x_coords.min():x_coords.max():100j, y_coords.min():y_coords.max():100j]
        
            grid_z = griddata((x_coords, y_coords), z_values, (grid_x, grid_y), method='cubic')
        
            # Build 3D surface plot
            fig_stress_heatmap_3d = go.Figure(data=[
                go.Surface( z=grid_z, x=grid_x, y=grid_y, colorscale='Hot', colorbar=dict(title='🔥 <b>Composite Stress Score</b>', x=1.0),
                    cmin=z_values.min(), cmax=z_values.max() )
            ])
        
            # Layout with centered bold title
            fig_stress_heatmap_3d.update_layout(
                title=dict( text='⚠️ <b>3D Heatmap Surface Plot of System Stress</b>', x=0.5, xanchor='center', yanchor='top' ),
                scene=dict( xaxis_title='⚙️ Total System Load', yaxis_title='📊 7-Day Rolling Std Dev Load', zaxis_title='🔥 Composite Stress Score' ), 
                autosize=True
            )
        
            # Render chart in Streamlit
            st.plotly_chart(fig_stress_heatmap_3d, width='stretch', key="stress_heatmap_3d")
        
            # Dynamic expander with stats
            with st.expander("ℹ️ More Information & Insights"):
                avg_stress = df_plot['Stress_Score'].mean()
                max_stress = df_plot['Stress_Score'].max()
                busiest_day = df_plot['Stress_Score'].idxmax().date()
                st.info(f"ℹ️ This heatmap visualizes composite stress across load and variability dimensions.")
                st.success(f"✅ Average Stress Score: {avg_stress:.2f}, Maximum Stress Score: {max_stress:.2f} on 📅 {busiest_day}.")
                st.warning("⚠️ High stress zones indicate periods where system load, variability, and intake factors combine to create operational strain.")
        
        except Exception as e:
            st.error(f"❌ Error rendering Stress Heatmap 3D chart: {e}")
        
        try:
            # Ensure required columns exist and handle NaNs
            required_pressure_cols = ['Net Daily Intake', 'Care Load Growth Rate', 'Sustained Positive Net Intake', 'High Load']
            df_plot_pressure = df.dropna(subset=required_pressure_cols).copy()
        
            # Normalize Net Daily Intake
            min_net_intake = df_plot_pressure['Net Daily Intake'].min()
            max_net_intake = df_plot_pressure['Net Daily Intake'].max()
            if (max_net_intake - min_net_intake) == 0:
                df_plot_pressure['Normalized_Net_Daily_Intake'] = 0.5
            else:
                df_plot_pressure['Normalized_Net_Daily_Intake'] = (
                    (df_plot_pressure['Net Daily Intake'] - min_net_intake) /
                    (max_net_intake - min_net_intake)
                )
        
            # Base Pressure Score
            df_plot_pressure['Pressure_Score'] = df_plot_pressure['Normalized_Net_Daily_Intake']
        
            # Growth Rate Multiplier
            df_plot_pressure['Care Load Growth Rate'] = df_plot_pressure['Care Load Growth Rate'].replace([np.inf, -np.inf], np.nan).fillna(0)
            df_plot_pressure['Positive_Growth_Rate_Multiplier'] = (df_plot_pressure['Care Load Growth Rate'].apply(lambda x: max(0, x)) / 100)
            df_plot_pressure['Pressure_Score'] *= (1 + df_plot_pressure['Positive_Growth_Rate_Multiplier'])
        
            # Sustained Intake Multiplier
            df_plot_pressure['Pressure_Score'] *= (1 + df_plot_pressure['Sustained Positive Net Intake'])
        
            # High Load Multiplier
            df_plot_pressure['Pressure_Score'] *= (1 + df_plot_pressure['High Load'])
        
            # Prepare data for surface plot
            x_coords = df_plot_pressure['Net Daily Intake'].values
            y_coords = df_plot_pressure['Care Load Growth Rate'].values
            z_values = df_plot_pressure['Pressure_Score'].values
        
            x_finite = x_coords[np.isfinite(x_coords)]
            y_finite = y_coords[np.isfinite(y_coords)]
        
            grid_x, grid_y = np.mgrid[
                x_finite.min():x_finite.max():100j,
                y_finite.min():y_finite.max():100j
            ]
        
            grid_z = griddata((x_coords, y_coords), z_values, (grid_x, grid_y), method='cubic')
        
            # Build 3D surface plot
            fig_pressure = go.Figure(data=[
                go.Surface( z=grid_z, x=grid_x, y=grid_y, colorscale='Plasma', colorbar=dict(title='💡 <b>Composite Pressure Score</b>', x=1.0),
                    cmin=z_values.min(), cmax=z_values.max()
                )
            ])
        
            # Layout with centered bold title
            fig_pressure.update_layout(
                title=dict( text='⚠️ <b>3D Heatmap Surface Plot of System Pressure</b>', x=0.5, xanchor='center', yanchor='top'),
                scene=dict( xaxis_title='🍽 Net Daily Intake', yaxis_title='📈 Care Load Growth Rate (%)', zaxis_title='💡 Composite Pressure Score'),
                autosize=True
            )
        
            # Render chart in Streamlit
            st.plotly_chart(fig_pressure, width='stretch', key="pressure_heatmap_3d")
        
            # Dynamic expander with stats
            with st.expander("ℹ️ More Information & Insights"):
                avg_pressure = df_plot_pressure['Pressure_Score'].mean()
                max_pressure = df_plot_pressure['Pressure_Score'].max()
                busiest_day = df_plot_pressure['Pressure_Score'].idxmax().date()
                st.info(f"ℹ️ This heatmap visualizes composite system pressure across intake and growth rate dimensions.")
                st.success(f"✅ Average Pressure Score: {avg_pressure:.2f}, Maximum Pressure Score: {max_pressure:.2f} on 📅 {busiest_day}.")
                st.warning("⚠️ High pressure zones indicate periods where intake, growth rate, and load factors combine to create operational strain.")
        
        except Exception as e:
            st.error(f"❌ Error rendering Pressure Heatmap 3D chart: {e}")

# -------------------------
# Recommendational Forecast Tab
# -------------------------
with tab_reco:
    st.header("🎯 Recommendational Forecast — Feature Engineering, Modeling & Recommendations")
    st.snow()

    st.markdown(
        """
        This section focuses on **predictive modeling and actionable insights**.  
        It builds upon the structural analysis by applying advanced techniques to forecast outcomes and recommend strategies.  
        """
    )
    
    st.caption("📐 **Feature Engineering** — Create lag variables, rolling statistics, and domain-specific features to improve model accuracy.")
    st.caption("🦾 **Modeling** — Train regression and classification models to evaluate performance under different scenarios.")
    st.caption("📊 **Evaluation** — Compare baseline vs engineered features, visualize accuracy, and assess volatility.")
    st.caption("🚨 **Alert Analysis** — Detect anomalies, threshold breaches, and trigger system alerts for proactive monitoring.")
    st.caption("💨 **Velocity Insights** — Measure rate-of-change dynamics to capture acceleration or deceleration in system load.")
    st.caption("🔮 **Future Forecasts** — Generate forward-looking predictions and provide decision-ready recommendations.")
    
    # -------------------------
    # Top KPIs — Recommendational Forecasts
    # -------------------------
    st.markdown("### 📊 Top Key Performance Indicators (KPIs)")
    
    df['Inflow_Velocity'] = df['Children transferred out of CBP custody'].rolling(window=3).mean() - df['Children transferred out of CBP custody'].rolling(window=10).mean()
    df['Operational_Pressure_Index'] = (df['Net Daily Intake'].rolling(window=7).mean() / (df['7-Day Rolling Std Dev Load'] + 1)).fillna(0)
    opi_mean = df['Operational_Pressure_Index'].mean()
    opi_std = df['Operational_Pressure_Index'].std()
    df['High_Pressure_Alert'] = (df['Operational_Pressure_Index'] > (opi_mean + opi_std)).astype(int)
    
    c1, c2, c3 = st.columns(3)
    
    # First row of KPIs
    c1.metric("📈 Current OPI", f"{df['Operational_Pressure_Index'].iloc[-1]:.2f}")
    if len(df) >= 2:
        latest_val = df['Inflow_Velocity'].iloc[-1]
        prev_val   = df['Inflow_Velocity'].iloc[-2]
        delta_val  = latest_val - prev_val
        c2.metric("⚡ Inflow Velocity Rate", f"{latest_val:.2f}", f"{delta_val:.2f}")
    elif len(df) == 1:
        latest_val = df['Inflow_Velocity'].iloc[-1]
        c2.metric("⚡ Inflow Velocity Rate", f"{latest_val:.2f}", "0")
    else:
        c2.metric("⚡ Inflow Velocity Rate", "N/A", "N/A")
    c3.metric("🚨 High Pressure Alerts", int(df['High_Pressure_Alert'].sum()))
    
    r_features, r_models, r_forecasts, r_future = st.tabs(["🧩 Feature Engineering", "🧪 Modeling & Evaluation", "🚨 Alerts & Velocity", "🔮 Future Predictions"])
        
    df_ml = df.copy()
    # -------------------------
    # Feature Engineering 
    # -------------------------
    with r_features:
        st.subheader("🔗 Feature Engineering")
        st.info("🧩 Lags, Rolling Statistics, & Date-based Features")
        st.markdown("### Lag Correlation")
        
        try:
            # Create lag features dynamically
            for i in range(1, 8):
                df_ml[f'Load_Lag_{i}d'] = df_ml['Total System Load'].shift(i)
            # Compute lag correlations
            lags = [f'Load_Lag_{i}d' for i in range(1, 8)]
            correlations = [df_ml['Total System Load'].corr(df_ml[lag]) for lag in lags]
    
            # Create interactive bar chart
            fig_lag_decay = px.bar( x=list(range(1, 8)), y=correlations, labels={'x': 'Lag Period (Days)', 'y': 'Pearson Correlation'},
                title='Predictive Power Decay: Lag Correlation with Current Load', color=correlations, color_continuous_scale='Viridis')
    
            # Adjust y-axis limits dynamically
            fig_lag_decay.update_yaxes(range=[min(correlations) - 0.05, 1.0])
    
            # Layout tweaks
            fig_lag_decay.update_layout(autosize=True, xaxis=dict(tickmode='linear'), title=dict(x=0.5, xanchor='center') )
    
            st.plotly_chart(fig_lag_decay, width='stretch')
    
            # Dynamic expander with summary stats
            with st.expander("Lag Correlation Insights"):
                st.info(f"🔹 Average correlation across lags: {sum(correlations)/len(correlations):.3f}")
                st.success(f"🔹 Strongest correlation at lag {lags[correlations.index(max(correlations))]}: {max(correlations):.3f}")
                st.warning(f"🔹 Weakest correlation at lag {lags[correlations.index(min(correlations))]}: {min(correlations):.3f}")
                
            avg_corr = sum(correlations)/len(correlations)
            if avg_corr < 0.3:
                st.toast("⚠️ Lag features show weak predictive power — consider adding external drivers.", icon="⚠️")
            else:
                st.toast("✅ Lag features provide strong predictive signals.", icon="📈")
                
            # Lags and rolling features
            for lag in range(1, 15):
                df_ml[f'lag_{lag}'] = df_ml['Total System Load'].shift(lag)
            for w in [7, 14, 30]:
                df_ml[f'roll_mean_{w}'] = df_ml['Total System Load'].rolling(window=w).mean().shift(1)
                df_ml[f'roll_std_{w}'] = df_ml['Total System Load'].rolling(window=w).std().shift(1)
            # cyclical encodings
            df_ml['dow'] = df_ml.index.dayofweek
            df_ml['dow_sin'] = np.sin(2 * np.pi * df_ml['dow'] / 7)
            df_ml['dow_cos'] = np.cos(2 * np.pi * df_ml['dow'] / 7)
            df_ml['month'] = df_ml.index.month
            df_ml['month_sin'] = np.sin(2 * np.pi * df_ml['month'] / 12)
            df_ml['month_cos'] = np.cos(2 * np.pi * df_ml['month'] / 12)
            df_ml['is_weekend'] = (df_ml.index.dayofweek >= 5).astype(int)
            df_ml['net_x_weekend'] = df_ml['Net Daily Intake'] * df_ml['is_weekend']
            df_ml.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_ml.dropna(inplace=True)
    
            # Feature distributions and pairwise scatter for top features
            with st.expander("Feature distributions & pairwise scatter"):
                st.dataframe(df_ml.head(6), width='stretch')
                top_features = ['lag_1', 'lag_7', 'roll_mean_7', 'roll_std_7', 'Net Daily Intake']
                cols = st.columns(3)
                for i, f in enumerate(top_features):
                    fig_dist = px.histogram(df_ml, x=f, nbins=40, title=f"Distribution: {f}")
                    fig_dist.update_layout(autosize=True)
                    cols[i % 3].plotly_chart(fig_dist, width='stretch')
                with st.expander("More Information"):
                    st.info("Lags, rolling stats, cyclical encodings, and interactions added to capture temporal patterns.")
                    st.success("These features are commonly effective for short-term time-series forecasting.")
            
        except Exception as e:
            st.error(f"Error generating lag decay chart: {e}")
        
        # 2. Rolling Statistics: To capture trends and seasonality
        # Rolling mean and standard deviation for 'Total System Load'
        # Shift by 1 to prevent data leakage (rolling window should not include current day's target)
        for window in [7, 14, 30]: # Weekly, bi-weekly, monthly windows
            df_ml[f'Load_Rolling_Mean_{window}d'] = df_ml['Total System Load'].rolling(window=window).mean()#.shift(1) # To view past
            df_ml[f'Load_Rolling_Std_{window}d'] = df_ml['Total System Load'].rolling(window=window).std()#.shift(1) # To view past
        
        st.markdown("### 🔗 Feature Interaction Matrix")
        try:
            # Build 3D scatter plot
            fig_interaction_roll_stats = go.Figure(data=[go.Scatter3d(
                x=df_ml['Load_Rolling_Mean_7d'],
                y=df_ml['Load_Rolling_Std_7d'],
                z=df_ml['Total System Load'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=df_ml['Total System Load'],
                    colorscale='Viridis',
                    opacity=0.7,
                    colorbar=dict(title='Actual Load')
                )
            )])
    
            # Layout tweaks
            fig_interaction_roll_stats.update_layout(
                title=dict(text='<b>Feature Interaction: Rolling Mean vs. Volatility vs. Target', x=0.5, xanchor='center', yanchor='top'),
                scene=dict(xaxis_title='7d Rolling Mean', yaxis_title='7d Rolling Std Dev', zaxis_title='Current Load'), autosize=True
            )
    
            st.plotly_chart(fig_interaction_roll_stats, width='stretch')
    
            # Dynamic expander with styled insights
            with st.expander("Feature Interaction Insights"):
                st.info(f"ℹ️ Average 7d Rolling Mean: {df_ml['Load_Rolling_Mean_7d'].dropna().mean():.2f}")
                st.success(f"✅ Average 7d Rolling Std Dev: {df_ml['Load_Rolling_Std_7d'].dropna().mean():.2f}")
                st.warning(f"⚠️ Average Current Load: {df_ml['Total System Load'].mean():.2f}")
                st.success(f"🔹 Max Current Load observed: {df_ml['Total System Load'].max():.2f}")
                st.warning(f"🔹 Min Current Load observed: {df_ml['Total System Load'].min():.2f}")
    
        except Exception as e:
            st.error(f"Error generating 3D interaction chart: {e}")
        
        # 3. Date-based Features: To capture cyclical patterns
        df_ml['Day_of_Week'] = df_ml.index.dayofweek # Monday=0, Sunday=6
        df_ml['Day_of_Month'] = df_ml.index.day
        df_ml['Month'] = df_ml.index.month
        df_ml['Year'] = df_ml.index.year
        df_ml['Week_of_Year'] = df_ml.index.isocalendar().week.astype(int)
        df_ml['Is_Weekend'] = (df_ml.index.dayofweek >= 5).astype(int) # 1 for weekend, 0 for weekday
        st.markdown("### 🔄 Weekly ↔ Monthly Load Patterns")
        
        try:
            # --- Day of Week Distribution ---
            day_labels = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
            df_ml['Day_Label'] = df_ml['Day_of_Week'].map(day_labels).astype(str)
            # Explicit order for days
            day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
            fig_day = px.box(
                df_ml,
                x='Day_Label',
                y='Total System Load',
                points='all',
                color='Day_Label',
                title='Load Distribution by Day of Week',
                labels={'Day_Label': 'Day of Week', 'Total System Load': 'Total System Load'},
                category_orders={'Day_Label': day_order},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_day.update_layout(autosize=True,
                title=dict(x=0.5, xanchor='center', yanchor="top")
            )
            st.plotly_chart(fig_day, width='stretch', key="day_distribution")
    
            with st.expander("Day of Week Insights"):
                st.info(f"ℹ️ Highest average load day: {df_ml.groupby('Day_Label')['Total System Load'].mean().idxmax()}")
                st.success(f"✅ Lowest average load day: {df_ml.groupby('Day_Label')['Total System Load'].mean().idxmin()}")
                st.warning(f"⚠️ Overall average load across days: {df_ml['Total System Load'].mean():.2f}")
    
            # --- Monthly Distribution ---
            df_ml['Month'] = df_ml['Month'].astype(int)
            month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            df_ml['Month_Label'] = df_ml['Month'].map(month_labels)
            month_order = list(month_labels.values())
    
            fig_month = px.box(
                df_ml,
                x='Month_Label',
                y='Total System Load',
                points='all',
                color='Month_Label',
                title='Monthly Seasonality Patterns',
                labels={'Month_Label': 'Month', 'Total System Load': 'Total System Load'},
                category_orders={'Month_Label': month_order},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_month.update_layout(autosize=True,
                title=dict(x=0.5, xanchor='center', yanchor="top")
            )
            st.plotly_chart(fig_month, width='stretch', key="month_distribution")
    
            with st.expander("Monthly Insights"):
                st.info(f"ℹ️ Peak average load month: {df_ml.groupby('Month_Label')['Total System Load'].mean().idxmax()}")
                st.success(f"✅ Lowest average load month: {df_ml.groupby('Month_Label')['Total System Load'].mean().idxmin()}")
                st.warning(f"⚠️ Overall average load across months: {df_ml['Total System Load'].mean():.2f}")
    
        except Exception as e:
            st.error(f"Error generating seasonality charts: {e}")

    # -------------------------
    # Modeling & Evaluation
    # -------------------------
    with r_models:
        st.subheader("⚙️ Modeling & Evaluation")
        st.info("🧪 Train regression and classification models to view their visualizations for best insights.")
    
        # -------------------------
        # Regression Setup
        # -------------------------
        target_variable = 'Total System Load'
        features = [col for col in df_ml.columns if col not in [target_variable, 'Load_Category', 'Day_Label', 'Month_Label']]
    
        # --- Handle NaNs globally ---
        imputer = SimpleImputer(strategy="mean")
        X = df_ml[features]
        y = df_ml[target_variable]
    
        try:
            # Split data (70/30, chronological)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
        
            # Train regression models
            figures_reg, performance_df_reg = train_and_evaluate_models(X_train, y_train, X_test, y_test)
            st.caption(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
            # -------------------------
            # Regression Models UI
            # -------------------------
            st.subheader("📈 Regression Models")
            
            model_list = [m for m in figures_reg.keys() if m != "Performance Heatmap"]
            selected_model = st.selectbox("🔍 Choose Regression Model:", model_list)
            st.plotly_chart(figures_reg[selected_model], width='stretch', key=f"{selected_model}_chart")
            
            with st.expander("📑 Regression Model Insights"):
                mae, rmse = performance_df_reg.at[selected_model, "MAE"], performance_df_reg.at[selected_model, "RMSE"]
                st.info(f"ℹ️ Selected model: **{selected_model}**")
                st.success(f"✅ MAE: {mae:.2f}")
                st.warning(f"⚠️ RMSE: {rmse:.2f}")
            
            # -------------------------
            # Performance Comparison Visualizations
            # -------------------------
            st.subheader("📊 Regression Performance Comparison")
            
            # Build extra figures once
            df_perf = performance_df_reg.reset_index()
            figures_reg.setdefault("3D Scatter", px.scatter_3d(
                df_perf, x="MAE", y="RMSE", z="index",
                color="index", hover_name="index",
                title="3D Scatter: Regression Performance"
            ))
            figures_reg.setdefault("3D Surface", go.Figure(data=[go.Surface(
                z=[[mae] for mae in df_perf["MAE"]]
            )]).update_layout(
                title="3D Surface: Regression Performance Landscape", autosize=True,
                scene=dict(xaxis_title="Model Index", yaxis_title="Metric Axis", zaxis_title="MAE")
            ))
            
            viz_option = st.selectbox("🎨 Choose Regression Visualization:", ["Heatmap", "3D Scatter", "3D Surface"])
            
            st.plotly_chart(figures_reg[viz_option if viz_option != "Heatmap" else "Performance Heatmap"],
                            width='stretch', key=f"{viz_option}_chart")
            
            with st.expander("Regression Performance Insights"):
                best_model = performance_df_reg["MAE"].idxmin()
                worst_model = performance_df_reg["MAE"].idxmax()
                st.success(f"🏆 Best model by MAE: **{best_model}** ({performance_df_reg['MAE'].min():.2f})")
                st.warning(f"⚠️ Worst model by MAE: **{worst_model}** ({performance_df_reg['MAE'].max():.2f})")
                st.info(f"ℹ️ {viz_option} shows all regression models together for quick comparison.")
            
            # -------------------------
            # Toast Alerts
            # -------------------------
            if performance_df_reg['MAE'].min() > 500:
                st.toast("⚠️ Regression models show high error.", icon="⚠️")
            else:
                st.toast("✅ Regression models performing within acceptable error bounds.", icon="📊")
        except Exception as e:
            st.warning(f"Not enoughh data for regression: {e}")
    
        # -------------------------
        # Classification Setup
        # -------------------------
        high_load_threshold = df_ml['Total System Load'].quantile(0.75)
        low_load_threshold = df_ml['Total System Load'].quantile(0.25)
    
        def get_load_category(load):
            if load >= high_load_threshold:
                return 'High'
            elif load <= low_load_threshold:
                return 'Low'
            else:
                return 'Medium'
    
        df_ml['Load_Category'] = df_ml['Total System Load'].apply(get_load_category)
        target_variable_clf = 'Load_Category'
        features_clf = [col for col in df_ml.columns if col not in [target_variable_clf, 'Total System Load', 'Day_Label', 'Month_Label']]
    
        X_clf = df_ml[features_clf]
        y_clf = df_ml[target_variable_clf].astype(str)
        label_encoder = LabelEncoder()
        y_clf_encoded = label_encoder.fit_transform(y_clf.values.ravel())
    
        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X_clf, y_clf_encoded, test_size=0.3, shuffle=True, stratify=y_clf_encoded, random_state=42
        )

        imputer = SimpleImputer(strategy="mean")
        X_train_clf_imputed = imputer.fit_transform(X_train_clf)
        X_test_clf_imputed = imputer.transform(X_test_clf)
    
        scaler_clf = StandardScaler()
        X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf_imputed)
        X_test_clf_scaled = scaler_clf.transform(X_test_clf_imputed)
        X_train_clf_scaled = pd.DataFrame(np.asarray(X_train_clf_scaled), columns=X_train_clf.columns, index=X_train_clf.index)
        X_test_clf_scaled = pd.DataFrame(np.asarray(X_test_clf_scaled), columns=X_test_clf.columns, index=X_test_clf.index)
        
        # -------------------------
        # Classification Models UI
        # -------------------------
        st.subheader("🧮 Classification Models")
    
        figures_clf, performance_df_clf = train_and_evaluate_classifiers(
            X_train_clf_scaled, y_train_clf, X_test_clf_scaled, y_test_clf, label_encoder
        )
    
        st.markdown("### 📊 Classification: Load Category (Low / Medium / High)")
        # 📦 Load category distribution with expander
        st.markdown("##### 📦 Load Category Distribution")
        st.bar_chart(df_ml['Load_Category'].value_counts())
        st.info("ℹ️ This chart shows how many days fall into each load category.")
        with st.expander("🔎 How Categories Were Defined"):
            st.success(f"🏆 High Load: Days where 'Total System Load' ≥ 75th percentile ({high_load_threshold:.2f})")
            st.warning(f"⚠️ Low Load: Days where 'Total System Load' ≤ 25th percentile ({low_load_threshold:.2f})")
            st.info("ℹ️ Medium Load: All days between the 25th and 75th percentile thresholds")
    
        st.subheader("📊 Classification Performance Comparison")
        selected_viz = st.selectbox("🔍 Choose Classification Visualization:", list(figures_clf.keys()))
        figures_clf[selected_viz].update_layout(autosize=True)
        st.plotly_chart(figures_clf[selected_viz], width='stretch')
    
        with st.expander("📑 Classification Performance Insights"):
            best_model = performance_df_clf.sort_values(
                by=['F1-Score','Accuracy','Recall'], ascending=[False,False,False]
            ).iloc[0]
            worst_model = performance_df_clf.sort_values(
                by=['F1-Score','Accuracy','Recall'], ascending=[True,True,True]
            ).iloc[0]
            st.success(f"🏆 Best Classification Model: **{best_model['Model']}** "
                    f"(F1={best_model['F1-Score']:.3f}, Acc={best_model['Accuracy']:.3f})")
            st.warning(f"⚠️ Worst Classification Model: **{worst_model['Model']}** "
                    f"(F1={worst_model['F1-Score']:.3f}, Acc={worst_model['Accuracy']:.3f})")
            st.info(f"ℹ️ {selected_viz} shows all classification models together for quick comparison.")
    
    # -------------------------
    # Forecasts & Recommendations
    # -------------------------
    with r_forecasts:
        st.subheader("🚨 Alert & Velocity")
        st.info("🔮 Tracking OPI Trends + Strain Events, Inflow Velocity (%) and Pressure-Stress System Load Dynamics.")
        st.subheader("📊 Operational Pressure Index (OPI)")
    
        try:
            required_cols = [
                'Children transferred out of CBP custody',
                'Net Daily Intake',
                '7-Day Rolling Std Dev Load'
            ]
            if all(col in df.columns for col in required_cols) and not df.empty:
                # Inflow velocity with NaN handling
                df['Inflow_Velocity'] = (
                    df['Children transferred out of CBP custody'].rolling(window=3).mean()
                    - df['Children transferred out of CBP custody'].rolling(window=10).mean()
                ).fillna(0)
    
                df['Operational_Pressure_Index'] = (
                    df['Net Daily Intake'].rolling(window=7).mean()
                    / (df['7-Day Rolling Std Dev Load'] + 1)
                ).fillna(0)
    
                opi_mean = df['Operational_Pressure_Index'].mean()
                opi_std = df['Operational_Pressure_Index'].std()
                df['High_Pressure_Alert'] = (df['Operational_Pressure_Index'] > (opi_mean + opi_std)).astype(int)
    
                # Build the figure
                fig_opi = go.Figure()
    
                fig_opi.add_trace(go.Scatter(
                    x=df.index,
                    y=df['Operational_Pressure_Index'],
                    mode='lines',
                    name='Operational Pressure Index',
                    line=dict(color='crimson')
                ))
    
                fig_opi.add_hline(
                    y=opi_mean + opi_std,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Alert Threshold ({opi_mean + opi_std:.2f})",
                    annotation_position="top left",
                    name='Alert Threshold'
                )
    
                y_fill = df['Operational_Pressure_Index'].where(df['High_Pressure_Alert'] == 1, np.nan)
    
                fig_opi.add_trace(go.Scatter(
                    x=df.index,
                    y=y_fill,
                    mode='lines',
                    fill='tozeroy',
                    fillcolor='rgba(255, 0, 0, 0.3)',
                    line=dict(color='rgba(255, 0, 0, 0)'),
                    name='Strain Event (Alert Period)'
                ))
    
                fig_opi.update_layout(
                    title=dict(text='<b>Operational Pressure Index (OPI) Over Time</b>', x=0.5, xanchor='center'),
                    legend=dict(title="📌 Scale", orientation="h", yanchor="bottom", y=-0.4, x=0.5, xanchor="center"),
                    xaxis_title='Date',
                    yaxis_title='Pressure Score',
                    autosize=True
                )
    
                st.plotly_chart(fig_opi, width='stretch')
    
                with st.expander("📑 OPI Insights"):
                    st.info("ℹ️ The Operational Pressure Index (OPI) measures intake velocity relative to discharge volatility.")
                    st.success(f"✅ Average OPI: {opi_mean:.2f}")
    
                with st.expander("🚨 Strain Event Details"):
                    strain_events = df[df['High_Pressure_Alert'] == 1]
                    st.info(f"📊 The alert threshold is set at **{opi_mean + opi_std:.2f}**.")
    
                    if not strain_events.empty:
                        st.warning("⚠️ Periods shaded in red indicate sustained high-pressure alerts.")
                        st.success(f"✅ Total strain events detected: **{len(strain_events)} days**.")
                        st.success(f"🏆 First Strain Event: {strain_events.index.min().strftime('%Y-%m-%d')}")
                        st.warning(f"⚠️ Most Recent Strain Event: {strain_events.index.max().strftime('%Y-%m-%d')}")
                    else:
                        st.info("✅ No strain events detected in the selected period.")
    
                # Toast for strain events
                if not strain_events.empty:
                    st.toast("🚨 Strain events detected! Check the red shaded periods in OPI's graph under Alert & Future Forecast's tab.", icon="⚠️")
                else:
                    st.toast("✅ No strain events detected in the selected period.", icon="✅")
            else:
                st.warning("⚠️ Not enough data or missing required columns to compute OPI.")
    
        except Exception as e:
            st.error(f"❌ Error rendering OPI chart: {e}")
    
        # -------------------------
        # 3D Pressure Surface (OPI vs Load vs Variability)
        # -------------------------
        st.markdown("### 🌐 3D Pressure Landscape (Load–Variability–OPI)")
        
        df_pressure = df.dropna(
            subset=['Operational_Pressure_Index', 'Total System Load', '7-Day Rolling Std Dev Load']
        ).copy()
        
        if not df_pressure.empty:
            x = df_pressure['Total System Load'].values
            y = df_pressure['7-Day Rolling Std Dev Load'].values
            z = df_pressure['Operational_Pressure_Index'].values
        
            try:
                grid_x, grid_y = np.mgrid[x.min():x.max():60j, y.min():y.max():60j]
                grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
        
                fig_pressure_landscape = go.Figure( data=[go.Surface(x=grid_x, y=grid_y, z=grid_z, colorscale='Inferno')] )
                fig_pressure_landscape.update_layout(
                    scene=dict(xaxis_title='Total System Load', yaxis_title='7-Day Rolling Std Dev Load', zaxis_title='Operational Pressure Index'),
                    autosize=True
                )
                st.plotly_chart(fig_pressure_landscape, width='stretch')
        
                # Dynamic insights
                with st.expander("📑 Pressure Landscape Insights"):
                    st.info("ℹ️ The pressure landscape shows where load and variability combine to create high OPI.")
                    st.success(f"✅ Current OPI: {df['Operational_Pressure_Index'].iloc[-1]:.2f}")
                    st.warning("⚠️ Use this landscape to define operational thresholds and trigger rules.")
        
                # Toast alerts based on OPI
                current_opi = df['Operational_Pressure_Index'].iloc[-1]
                if current_opi > (df['Operational_Pressure_Index'].mean() + df['Operational_Pressure_Index'].std()):
                    st.toast("🚨 Current OPI exceeds alert threshold!", icon="🔥")
                else:
                    st.toast("✅ Current OPI is within safe limits.", icon="📈")
        
            except Exception:
                st.error("❌ Unable to interpolate pressure landscape for current data slice.")
                st.toast("⚠️ Pressure landscape interpolation failed.", icon="⚠️")
        
        else:
            st.info("ℹ️ Insufficient data or 3D disabled for pressure landscape.")
            st.toast("⚠️ No data available for 3D pressure surface.", icon="⚠️")
        
        # -------------------------
        # Dynamic Streamlit chart for Inflow Velocity
        # -------------------------
        st.markdown("### 🌊 Inflow Velocity Over Time")
        
        fig_inflow_velocity = px.line( df, x=df.index, y='Inflow_Velocity', title='<b>Inflow Velocity Over Time</b>', 
            labels={'Inflow_Velocity': 'Inflow Velocity', 'index': 'Date'}, line_shape='linear' )
        
        fig_inflow_velocity.update_layout( xaxis_title='Date', yaxis_title='Inflow Velocity', autosize=True, title=dict(x=0.5, xanchor='center') )
        
        st.plotly_chart(fig_inflow_velocity, width='stretch')
        
        # -------------------------
        # Insights Expander
        # -------------------------
        with st.expander("📑 Inflow Velocity Insights"):
            st.info("ℹ️ Inflow Velocity measures the short-term intake speed relative to longer-term discharge trends.")
            
            latest_inflow_value = df['Inflow_Velocity'].iloc[-1]
            latest_inflow_date = df.index[-1].strftime("%Y-%m-%d")
            
            st.success(f"🌊 Latest Inflow Velocity: {latest_inflow_value:.2f} 📅 {latest_inflow_date}")
            
            # Warning threshold (mean + std)
            inflow_threshold = df['Inflow_Velocity'].mean() + df['Inflow_Velocity'].std()
            if latest_inflow_value > inflow_threshold:
                st.warning(f"⚠️ Current inflow velocity ({latest_inflow_value:.2f}) exceeds the alert threshold ({inflow_threshold:.2f}).")
                st.toast(f"🚨 Inflow velocity spike detected on 📅 {latest_inflow_date}!", icon="🔥")
            else:
                st.info(f"✅ Current inflow velocity ({latest_inflow_value:.2f}) is within safe limits.")
                st.toast(f"📈 Inflow velocity stable on 📅 {latest_inflow_date}.", icon="📈")
        
        # -------------------------
        # Dynamic Streamlit 3D Scatter for Inflow Velocity
        # -------------------------
        st.markdown("### 🌀 Tri‑Axis Pressure-Stress System Dynamics")
        
        # Ensure relevant columns are not NaN for plotting
        df_3d_inflow = df.dropna(subset=['Inflow_Velocity', 'Operational_Pressure_Index', 'Total System Load']).copy()
        
        if not df_3d_inflow.empty:
            fig_inflow_3d = go.Figure(data=[
                go.Scatter3d(
                    x=df_3d_inflow['Operational_Pressure_Index'],
                    y=df_3d_inflow['Total System Load'],
                    z=df_3d_inflow['Inflow_Velocity'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=df_3d_inflow['Inflow_Velocity'],  # Color by Inflow Velocity
                        colorscale='Jet',
                        opacity=0.8,
                        colorbar=dict(title='<b>Inflow Velocity</b>', x=1.0)
                    ),
                    text=[
                        f"📅 {date.strftime('%Y-%m-%d')}<br>OPI: {opi:.2f}<br>Total Load: {tsl:.0f}<br>Inflow Velocity: {iv:.2f}"
                        for date, opi, tsl, iv in zip(
                            df_3d_inflow.index,
                            df_3d_inflow['Operational_Pressure_Index'],
                            df_3d_inflow['Total System Load'],
                            df_3d_inflow['Inflow_Velocity']
                        )
                    ],
                    hoverinfo='text'
                )
            ])
        
            fig_inflow_3d.update_layout(
                scene=dict( xaxis_title='Operational Pressure Index', yaxis_title='Total System Load', zaxis_title='Inflow Velocity' ),
                title=dict( text='<b>3D Scatter Plot: Inflow Velocity vs. Operational Pressure & Total Load</b>', x=0.5, xanchor='center' ),
                autosize=True
            )
        
            st.plotly_chart(fig_inflow_3d, width='stretch')
        
            # -------------------------
            # Insights Expander
            # -------------------------
            with st.expander("📑 3D Pressure–Stress Insights"):
                latest_iv = df_3d_inflow['Inflow_Velocity'].iloc[-1]
                latest_opi = df_3d_inflow['Operational_Pressure_Index'].iloc[-1]
                latest_load = df_3d_inflow['Total System Load'].iloc[-1]
                latest_date = df_3d_inflow.index[-1].strftime("%Y-%m-%d")
            
                st.info("ℹ️ This 3D scatter visualizes how **inflow velocity** interacts with the **Operational Pressure Index (OPI)** and **Total System Load**, highlighting stress dynamics in the system.")
                
                st.success(
                    f"📅 {latest_date} | 🌊 Inflow Velocity: {latest_iv:.2f} | "
                    f"📈 Total Load: {latest_load:.0f} | 🔧 OPI: {latest_opi:.2f}"
                )
            
                # Threshold warning based on inflow velocity relative to system stress
                inflow_threshold = df_3d_inflow['Inflow_Velocity'].mean() + df_3d_inflow['Inflow_Velocity'].std()
                if latest_iv > inflow_threshold:
                    st.warning(
                        f"⚠️ Stress Alert: Inflow velocity ({latest_iv:.2f}) is above the system’s safe threshold ({inflow_threshold:.2f}), "
                        f"indicating potential overload pressure."
                    )
                    st.toast(f"🚨 System stress spike detected on 📅 {latest_date}!", icon="🔥")
                else:
                    st.info(
                        f"✅ Current inflow velocity ({latest_iv:.2f}) remains within safe limits, "
                        f"suggesting balanced intake relative to system load."
                    )
                    st.toast(f"📈 System stress stable on 📅 {latest_date}.", icon="📈")
            
        else:
            st.info("ℹ️ Insufficient data for 3D inflow scatter plot.")
            st.toast("⚠️ No data available for 3D inflow visualization.", icon="⚠️")
        
        # -------------------------
        # Final Pressure & Stress Insights
        # -------------------------
        st.markdown("**📊 Pressure & Stress Insights**")
        p1, p2, p3 = st.columns(3)
        p1.metric("Current OPI", f"{df['Operational_Pressure_Index'].iloc[-1]:.2f}")
        p2.metric("High Pressure Alerts", int(df['High_Pressure_Alert'].sum()))
        p3.metric("Recent Inflow Velocity", f"{df['Inflow_Velocity'].iloc[-1]:.1f}")
        if len(df) >= 2:
            inflow_rate = df['Inflow_Velocity'].iloc[-1] - df['Inflow_Velocity'].iloc[-2]
            p1.metric("Inflow Velocity Rate", f"{df['Inflow_Velocity'].iloc[-1]:.2f}", delta=f"{inflow_rate:.2f}")
        elif len(df) == 1:
            latest_val = df['Inflow_Velocity'].iloc[-1]
            p1.metric("Inflow Velocity Rate", f"{latest_val:.2f}", delta="0")
        else:
            p1.metric("Inflow Velocity Rate", "N/A", delta="N/A")
            
        # Pressure–Stress Coupling Index (OPI × Inflow / Variability)
        coupling_index = (
            df['Operational_Pressure_Index'].iloc[-1] * df['Inflow_Velocity'].iloc[-1]
        ) / (df['7-Day Rolling Std Dev Load'].iloc[-1] + 1)
        p2.metric("Pressure–Stress Coupling Index", f"{coupling_index:.2f}")
        
        # Toast for high pressure alerts
        if df['High_Pressure_Alert'].sum() > 0:
            st.toast(f"🚨 {df['High_Pressure_Alert'].sum()} high-pressure alerts detected!", icon="⚠️")
        else:
            st.toast("✅ No high-pressure alerts detected.", icon="✅")
        
    with r_future:
        st.subheader("🔮 Future Forecasting (GBR Model)")
        # -------------------------
        # User control: prediction horizon
        # -------------------------
        months_to_predict = st.slider(
            "Select prediction horizon (months)",
            min_value=0,
            max_value=24,
            value=12,  # default 12 months
            step=1, key="rolling_window_slider"  
        )
        
        # Convert months to days (approximate 30.4 days per month)
        future_steps = int(months_to_predict * 30.4)
        st.info(f"📊 Forecasting for the next **{months_to_predict} months** (~{future_steps} days).")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("📈 Total System Load — the overall number of children currently in HHS care, representing the cumulative demand placed on the system.")
        with col2:
            st.info("🌊 Inflow Velocity — the rate at which children are transferred out of CBP custody, measured as short‑term intake speed relative to longer‑term discharge trends.")
        
        if months_to_predict >= 18:
            st.toast("⚠️ Long forecast horizon — reliability decreases.", icon="⚠️")
        elif months_to_predict <= 6:
            st.toast("✅ Short horizon — forecasts are stable.", icon="📈")
        
        # Add cyclical features
        df_ml['Day_of_Week_sin'] = np.sin(2 * np.pi * df_ml['Day_of_Week'] / 7)
        df_ml['Day_of_Week_cos'] = np.cos(2 * np.pi * df_ml['Day_of_Week'] / 7)
        df_ml['Month_sin'] = np.sin(2 * np.pi * df_ml['Month'] / 12)
        df_ml['Month_cos'] = np.cos(2 * np.pi * df_ml['Month'] / 12)
        df_ml['Day_of_Month_sin'] = np.sin(2 * np.pi * df_ml['Day_of_Month'] / 31)
        df_ml['Day_of_Month_cos'] = np.cos(2 * np.pi * df_ml['Day_of_Month'] / 31)
        
        # -------------------------
        # Prepare dataset for re-training (date/exogenous features only)
        # -------------------------
        X_full = df_ml[['Day_of_Week','Day_of_Month','Month','Year','Week_of_Year','Is_Weekend',
                        'Day_of_Week_sin','Day_of_Week_cos','Month_sin','Month_cos',
                        'Day_of_Month_sin','Day_of_Month_cos']]
        
        y_full = df_ml['Total System Load']
        
        
        # Handle NaNs
        imputer = SimpleImputer(strategy="mean")
        try:
            X_full_imputed = imputer.fit_transform(X_full)
        
            # Scale features
            scaler_forecast = StandardScaler()
            X_full_scaled = scaler_forecast.fit_transform(X_full_imputed)
            
            # Train GBR model
            gbr_model_retrained = GradientBoostingRegressor(
                n_estimators=1000, learning_rate=1, max_depth=3, random_state=42
            )
            gbr_model_retrained.fit(X_full_scaled, y_full)
            
            # -------------------------
            # Build future feature matrix
            # -------------------------
            last_date = df_ml.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
            
            future_index = pd.DatetimeIndex(future_dates)
            future_X = pd.DataFrame(index=future_index)
                
            # Populate date-based features
            future_X['Day_of_Week'] = future_index.dayofweek
            future_X['Day_of_Month'] = future_index.day
            future_X['Month'] = future_index.month
            future_X['Year'] = future_index.year
            future_X['Week_of_Year'] = future_index.isocalendar().week.astype(int)
            future_X['Is_Weekend'] = (future_index.dayofweek >= 5).astype(int)
            
            # Cyclical features
            future_X['Day_of_Week_sin'] = np.sin(2 * np.pi * future_X['Day_of_Week'].astype(float) / 7)
            future_X['Day_of_Week_cos'] = np.cos(2 * np.pi * future_X['Day_of_Week'].astype(float) / 7)
            future_X['Month_sin'] = np.sin(2 * np.pi * future_X['Month'].astype(float) / 12)
            future_X['Month_cos'] = np.cos(2 * np.pi * future_X['Month'].astype(float) / 12)
            future_X['Day_of_Month_sin'] = np.sin(2 * np.pi * future_X['Day_of_Month'].astype(float) / 31)
            future_X['Day_of_Month_cos'] = np.cos(2 * np.pi * future_X['Day_of_Month'].astype(float) / 31)
            
            # -------------------------
            # Direct forecast (no iterative loop)
            # -------------------------
            future_X_imputed = imputer.transform(future_X)
            future_X_scaled = scaler_forecast.transform(future_X_imputed)
            future_forecast = gbr_model_retrained.predict(future_X_scaled)
            
            future_forecast = pd.Series(future_forecast, index=future_dates)
            future_forecast = future_forecast.clip(lower=0)
            
            # -------------------------
            # Alerting with st.toast
            # -------------------------
            # Example threshold for alerts
            alert_threshold = df['Operational_Pressure_Index'].mean() + df['Operational_Pressure_Index'].std()
        
            # Toast for forecast alerts
            if (future_forecast > alert_threshold).any():
                st.toast("⚠️ Forecasted system load exceeds the alert threshold!", icon="🔥")
            else:
                st.toast("✅ Forecasted system load remains within safe limits.", icon="📈")   
                
            # -------------------------
            # Forecast Inflow Velocity (reuse same future_X)
            # -------------------------
            df_ml['Inflow_Velocity'] = df['Inflow_Velocity']
            # Train inflow velocity model
            target_inflow = 'Inflow_Velocity'
            features_inflow = X_full.columns  # reuse same date/cyclical features
            
            X_inflow = df_ml[features_inflow].dropna()
            y_inflow = df_ml.loc[X_inflow.index, target_inflow]
            
            imputer_inflow = SimpleImputer(strategy="mean")
            X_inflow_imputed = imputer_inflow.fit_transform(X_inflow)
            
            scaler_inflow = StandardScaler()
            X_inflow_scaled = scaler_inflow.fit_transform(X_inflow_imputed)
            
            gbr_inflow = GradientBoostingRegressor(n_estimators=1000, learning_rate=1, max_depth=3, random_state=42)
            gbr_inflow.fit(X_inflow_scaled, y_inflow)
            
            # Forecast inflow velocity using same future_X
            future_X_inflow_imputed = imputer_inflow.transform(future_X)
            future_X_inflow_scaled = scaler_inflow.transform(future_X_inflow_imputed)
            future_forecast_inflow = gbr_inflow.predict(future_X_inflow_scaled)
            future_forecast_inflow = pd.Series(future_forecast_inflow, index=future_dates)
            
            # -------------------------
            # Plot both forecasts integrated
            # -------------------------
            fig = go.Figure()
            
            # Historical
            fig.add_trace(go.Scatter(
                x=df_ml.index, y=df_ml['Total System Load'],
                mode='lines', name='Historical Load', line=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=df_ml.index, y=df_ml['Inflow_Velocity'],
                mode='lines', name='Historical Inflow Velocity', line=dict(color='green')
            ))
            
            # Forecasts
            fig.add_trace(go.Scatter(
                x=future_forecast.index, y=future_forecast,
                mode='lines', name='Future Forecast Load (GBR)', line=dict(color='red', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=future_forecast_inflow.index, y=future_forecast_inflow,
                mode='lines', name='Future Forecast Inflow Velocity (GBR)', line=dict(color='orange', dash='dot')
            ))
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, x=0.5, xanchor="center"), autosize=True)
            
            st.plotly_chart(fig, width='stretch', key="Future_forecast_Load_+_Inflow_Velocity")
            
            with st.expander("📑 Forecast Insights"):
                st.info("ℹ️ Forecasts are generated using Gradient Boosting Regressors retrained on the full dataset with date-based and cyclical features ('Day_of_Week', 'Day_of_Month', 'Month', 'Year', 'Week_of_Year', 'Is_Weekend',  \
                        'Day_of_Week_sin', 'Day_of_Week_cos', 'Month_sin', 'Month_cos',  \
                        'Day_of_Month_sin', 'Day_of_Month_cos').")
                st.success(f"✅ Generated next {len(future_forecast)} days predictions for Total System Load and Inflow Velocity.")
                st.warning("⚠️ Reliability decreases for horizons beyond 18–24 months, as lag/rolling features are not suitable/included in this forecast approach.")
                # Latest forecasted values with dates
                latest_load_date = future_forecast.index[-1].strftime("%Y-%m-%d")
                latest_inflow_date = future_forecast_inflow.index[-1].strftime("%Y-%m-%d")
                
                st.success(f"📈 Latest forecasted Load: {future_forecast.iloc[-1]:.2f} on 📅 {latest_load_date}")
                st.success(f"🌊 Latest forecasted Inflow Velocity: {future_forecast_inflow.iloc[-1]:.2f} on 📅 {latest_inflow_date}")
            
            # -------------------------
            # Toast alert for inflow velocity threshold
            # -------------------------
            inflow_threshold = df_ml['Inflow_Velocity'].mean() + df_ml['Inflow_Velocity'].std()
            if future_forecast_inflow.iloc[-1] > inflow_threshold:
                st.toast(f"🚨 Latest forecasted inflow velocity ({future_forecast_inflow.iloc[-1]:.2f}) exceeds threshold {inflow_threshold:.2f}!", icon="🔥")
            else:
                st.toast(f"✅ Latest forecasted inflow velocity ({future_forecast_inflow.iloc[-1]:.2f}) is within safe limits.", icon="📈")
                
            
        except Exception as e:
            st.warning(f"Not enough data for fitting: {e}")
        
        # -------------------------
        # Footer
        # -------------------------
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: white; font-size: 14px; line-height: 1.6;'>
            
                🎶 In halls of data, echoes rise,
                a chorus of numbers beneath the skies.
                From custody’s gate to care’s embrace,
                each metric sings of a system’s pace.
                
                🎵 The charts are strings, the plots a drum,
                volatility hums where the children come.
                Backlogs whisper in minor key,
                while forecasts chant of what may be.
                
                🎼 So let the dashboard strike its chord,
                balancing burdens with insight stored.
                In harmony, trends and truths align,
                a symphony guiding care through time.
                
                🔖 Created & Powered by Prathamesh Bhurke
                
            </div>
            """,
            unsafe_allow_html=True
        )
        
        
        # Buttons for links
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col3:
            st.link_button("📂 GitHub Repository",
                        "https://github.com/Prathamesh666/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children")
        with col4:
            st.link_button("📑 Research Paper",
                        "https://prathamesh666.github.io/System-Capacity-Care-Load-Analytics-for-Unaccompanied-Children/Research%20Paper.html")