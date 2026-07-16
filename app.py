from pathlib import Path
from datetime import timedelta
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Blast Furnace Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "2.0.0-offline"
DEFAULT_FILE = Path(__file__).parent / "data" / "synthetic_bf_historian_7days.csv"

REQUIRED_COLUMNS = [
    "timestamp",
    "quality",
    "hot_blast_temperature_c",
    "hot_blast_flow_nm3_min",
    "blast_pressure_bar",
    "differential_pressure_bar",
    "pci_rate_kg_thm",
    "oxygen_enrichment_pct",
    "top_gas_temperature_c",
    "top_gas_co_pct",
    "top_gas_co2_pct",
    "stockline_m",
    "cooling_water_delta_t_c",
    "hot_metal_temperature_c",
    "hot_metal_silicon_pct",
]

LIMITS = {
    "hot_blast_temperature_c": {"low": 1050.0, "high": 1250.0, "label": "Hot blast temperature", "unit": "°C"},
    "hot_blast_flow_nm3_min": {"low": 3600.0, "high": 4600.0, "label": "Hot blast flow", "unit": "Nm³/min"},
    "blast_pressure_bar": {"low": 3.2, "high": 4.5, "label": "Blast pressure", "unit": "bar"},
    "differential_pressure_bar": {"low": 1.0, "high": 1.8, "label": "Differential pressure", "unit": "bar"},
    "pci_rate_kg_thm": {"low": 120.0, "high": 190.0, "label": "PCI rate", "unit": "kg/tHM"},
    "oxygen_enrichment_pct": {"low": 1.0, "high": 5.0, "label": "Oxygen enrichment", "unit": "%"},
    "top_gas_temperature_c": {"low": 90.0, "high": 180.0, "label": "Top gas temperature", "unit": "°C"},
    "top_gas_co_pct": {"low": 18.0, "high": 28.0, "label": "Top gas CO", "unit": "%"},
    "top_gas_co2_pct": {"low": 16.0, "high": 24.0, "label": "Top gas CO₂", "unit": "%"},
    "stockline_m": {"low": 1.2, "high": 2.5, "label": "Stockline", "unit": "m"},
    "cooling_water_delta_t_c": {"low": 4.0, "high": 11.0, "label": "Cooling-water ΔT", "unit": "°C"},
    "hot_metal_temperature_c": {"low": 1460.0, "high": 1520.0, "label": "Hot-metal temperature", "unit": "°C"},
    "hot_metal_silicon_pct": {"low": 0.30, "high": 0.80, "label": "Hot-metal silicon", "unit": "%"},
}

PALETTE = {
    "bg": "#f4f6f9",
    "panel": "#ffffff",
    "panel2": "#ffffff",
    "text": "#101f30",
    "muted": "#5a6e85",
    "green": "#198754",
    "amber": "#d97706",
    "red": "#dc3545",
    "blue": "#0d6efd",
    "purple": "#6f42c1",
}

# ============================================================
# VISUAL STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            180deg,
            #ffffff 0%,
            #f4f7fb 100%
        );
        color: #172033;
    }

    .block-container {

    .block-container h1:first-of-type {
        margin-top: 0 !important;
        margin-bottom: 0.55rem !important;
        padding-top: 0 !important;
        line-height: 1.15 !important;
    }

    [data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        background: transparent !important;
    }
[data-testid="stToolbar"] {
    top: 0.25rem !important;
    right: 0.75rem !important;
}
    [data-testid="stSidebar"] {
        background-color: #e8eff6 !important;
        border-right: 1px solid #bdcbd9 !important;
        box-shadow: 4px 0 14px rgba(30, 64, 105, 0.10) !important;
    }

    [data-testid="stSidebar"] * {
        color: #172033;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #17365d !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8e1ea;
        border-radius: 14px;
        min-height: 105px !important;
        height: 105px !important;
        padding: 0.65rem !important;
        overflow: visible !important;
        box-shadow: 0 5px 18px rgba(30, 64, 105, 0.08);
    }

     {
        color: #526477 !important;
        font-size: 0.80rem !important;
        line-height: 1.1 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stMetricValue"] {
        color: #172033 !important;
        font-size: 1.55rem !important;
        line-height: 1.05 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="stMetricValue"] > div {
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    [data-testid="column"] {
        min-width: 0 !important;
    }

    .state-card {
        min-height: 105px !important;
        height: 105px !important;
        padding: 0.4rem !important;
        font-size: 0.95rem !important;
        line-height: 1.1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
        text-align: center !important;
        border-radius: 14px !important;
    }
[data-testid="stMetricLabel"] {
    color: #334155 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    line-height: 1.15 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    opacity: 1 !important;
}

[data-testid="stMetricLabel"] p {
    color: #334155 !important;
    opacity: 1 !important;
}

[data-testid="stMetricLabel"] div {
    color: #334155 !important;
    opacity: 1 !important;
}

    .state-amber {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fcd34d;
    }

    .state-red {
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: 0.65rem !important;
    }

    div[data-testid="stTabs"] {
        margin-top: 0.25rem !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #172033;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

@st.cache_data(show_spinner=False)
def load_csv(source):
    df = pd.read_csv(source)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_timestamps = int(df["timestamp"].isna().sum())
    if bad_timestamps:
        raise ValueError(f"{bad_timestamps} timestamp values could not be parsed.")

    for column in REQUIRED_COLUMNS:
        if column not in ("timestamp", "quality"):
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["quality"] = df["quality"].fillna("UNKNOWN").astype(str).str.upper()
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return add_derived_columns(df)


def add_derived_columns(df):
    result = df.copy()
    denominator = (result["top_gas_co_pct"] + result["top_gas_co2_pct"]).replace(0, np.nan)
    result["gas_utilisation_pct"] = 100 * result["top_gas_co2_pct"] / denominator
    result["permeability_index"] = result["differential_pressure_bar"] / result["blast_pressure_bar"].clip(lower=0.1)
    result["raft_proxy_c"] = (
        2100
        + 0.55 * (result["hot_blast_temperature_c"] - 1000)
        + 18 * result["oxygen_enrichment_pct"]
        - 0.35 * (result["pci_rate_kg_thm"] - 120)
    )
    result["co_co2_ratio"] = result["top_gas_co_pct"] / result["top_gas_co2_pct"].replace(0, np.nan)
    result["specific_blast_proxy"] = result["hot_blast_flow_nm3_min"] / 8.0
    result["thermal_index"] = (
        0.45 * normalize(result["hot_metal_temperature_c"], 1440, 1540)
        + 0.30 * normalize(result["hot_blast_temperature_c"], 1000, 1300)
        + 0.25 * normalize(result["hot_metal_silicon_pct"], 0.2, 1.0)
    ) * 100
    return result


def normalize(series, low, high):
    return ((series - low) / (high - low)).clip(0, 1)


def data_quality_report(df):
    numeric = [c for c in REQUIRED_COLUMNS if c not in ("timestamp", "quality")]
    rows = []
    for column in numeric:
        series = df[column]
        limit = LIMITS.get(column, {})
        physical_low = limit.get("low", -np.inf)
        physical_high = limit.get("high", np.inf)
        rows.append({
            "Tag": column,
            "Missing": int(series.isna().sum()),
            "Missing %": round(100 * series.isna().mean(), 3),
            "Min": series.min(),
            "Max": series.max(),
            "Outside prototype band": int(((series < physical_low) | (series > physical_high)).sum()),
            "Flat-line points": flatline_count(series),
        })
    return pd.DataFrame(rows)


def flatline_count(series, window=10):
    clean = series.ffill()
    return int((clean.rolling(window).std() < 1e-9).sum())

# ============================================================
# TWIN, ALARM, FORECAST, AND SCENARIO ENGINES
# ============================================================

def evaluate_twin(row, previous=None):
    alarms = []
    score = 100

    def add_alarm(severity, subsystem, message, evidence, points):
        nonlocal score
        alarms.append({
            "Severity": severity,
            "Subsystem": subsystem,
            "Message": message,
            "Evidence": evidence,
        })
        score -= points

    if row["quality"] != "GOOD":
        add_alarm("HIGH", "Instrumentation", "Sensor quality is not GOOD", f"Quality={row['quality']}", 25)
    if pd.isna(row["top_gas_temperature_c"]):
        add_alarm("HIGH", "Top gas", "Top-gas temperature unavailable", "Value is missing", 20)
    if row["differential_pressure_bar"] > 1.8:
        add_alarm("CRITICAL", "Gas flow", "High furnace differential pressure", f"ΔP={row['differential_pressure_bar']:.2f} bar", 35)
    if row["permeability_index"] > 0.45:
        add_alarm("HIGH", "Gas flow", "Possible permeability deterioration", f"Index={row['permeability_index']:.3f}", 20)
    if row["hot_metal_temperature_c"] < 1460:
        add_alarm("HIGH", "Thermal", "Low hot-metal temperature indication", f"HM temp={row['hot_metal_temperature_c']:.1f} °C", 25)
    if row["hot_metal_temperature_c"] > 1520:
        add_alarm("MEDIUM", "Thermal", "High hot-metal temperature indication", f"HM temp={row['hot_metal_temperature_c']:.1f} °C", 18)
    if row["hot_metal_silicon_pct"] > 0.80:
        add_alarm("MEDIUM", "Chemistry", "High hot-metal silicon indication", f"Si={row['hot_metal_silicon_pct']:.3f} %", 12)
    if row["gas_utilisation_pct"] < 42:
        add_alarm("MEDIUM", "Top gas", "Low gas-utilisation proxy", f"ηCO={row['gas_utilisation_pct']:.1f} %", 12)
    if row["cooling_water_delta_t_c"] > 11:
        add_alarm("HIGH", "Cooling", "High cooling-water temperature rise", f"ΔT={row['cooling_water_delta_t_c']:.2f} °C", 22)
    if previous is not None:
        dp_rate = row["differential_pressure_bar"] - previous["differential_pressure_bar"]
        if dp_rate > 0.08:
            add_alarm("MEDIUM", "Gas flow", "Rapid differential-pressure rise", f"1-step increase={dp_rate:.3f} bar", 10)

    score = max(0, min(100, score))
    if row["differential_pressure_bar"] > 1.8 or row["permeability_index"] > 0.48:
        state, css = "HANGING RISK", "state-red"
    elif row["hot_metal_temperature_c"] < 1460:
        state, css = "COLD CONDITION", "state-amber"
    elif row["hot_metal_temperature_c"] > 1520:
        state, css = "HOT CONDITION", "state-amber"
    elif row["quality"] != "GOOD" or pd.isna(row["top_gas_temperature_c"]):
        state, css = "SENSOR DEGRADED", "state-amber"
    else:
        state, css = "STABLE", "state-green"

    return {"state": state, "css": css, "score": score, "alarms": alarms}


def forecast_temperature(history, horizon_minutes=30):
    """Transparent trend-plus-process proxy forecast; no trained production model."""
    clean = history.dropna(subset=["hot_metal_temperature_c"]).tail(120)
    if len(clean) < 10:
        return np.nan, np.nan, "INSUFFICIENT DATA"

    y = clean["hot_metal_temperature_c"].to_numpy()
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    recent = clean.iloc[-1]
    process_adjustment = (
        0.025 * (recent["hot_blast_temperature_c"] - 1160)
        - 0.04 * (recent["pci_rate_kg_thm"] - 155)
        + 0.8 * (recent["oxygen_enrichment_pct"] - 3)
        - 1.5 * (recent["differential_pressure_bar"] - 1.36)
    )
    forecast = y[-1] + slope * horizon_minutes + process_adjustment
    residual = y - (np.polyval(np.polyfit(x, y, 1), x))
    uncertainty = max(3.0, float(np.nanstd(residual)) * 1.96)
    confidence = "MEDIUM" if len(clean) >= 60 and uncertainty < 8 else "LOW"
    return forecast, uncertainty, confidence


def forecast_silicon(history, temperature_forecast):
    clean = history.dropna(subset=["hot_metal_silicon_pct"]).tail(120)
    if clean.empty or pd.isna(temperature_forecast):
        return np.nan
    recent = clean.iloc[-1]
    return max(0, recent["hot_metal_silicon_pct"] + 0.0025 * (temperature_forecast - recent["hot_metal_temperature_c"]))


def scenario_engine(row, hbt_delta, pci_delta, oxygen_delta, blast_delta, moisture_proxy):
    hm_delta = 0.10 * hbt_delta - 0.16 * pci_delta + 4.0 * oxygen_delta + 0.008 * blast_delta - 0.5 * moisture_proxy
    raft_delta = 0.55 * hbt_delta - 0.35 * pci_delta + 18 * oxygen_delta - 3.0 * moisture_proxy
    fuel_delta = pci_delta - 0.04 * hbt_delta - 1.5 * oxygen_delta
    productivity_delta_pct = 0.018 * blast_delta + 0.6 * oxygen_delta - 0.10 * max(0, pci_delta)
    gas_delta = 0.03 * hbt_delta - 0.04 * pci_delta + 0.2 * oxygen_delta
    return {
        "hm_temperature": row["hot_metal_temperature_c"] + hm_delta,
        "hm_delta": hm_delta,
        "raft": row["raft_proxy_c"] + raft_delta,
        "fuel_delta": fuel_delta,
        "productivity_delta_pct": productivity_delta_pct,
        "gas_utilisation": row["gas_utilisation_pct"] + gas_delta,
    }

# ============================================================
# CHART HELPERS
# ============================================================

def apply_chart_theme(fig, height=330):
    fig.update_layout(
        height=height,
        margin=dict(
            l=28,
            r=20,
            t=105,
            b=35,
        ),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(
            color="#172033",
            family="Arial, sans-serif",
        ),
        title=dict(
            x=0.01,
            xanchor="left",
            y=0.97,
            yanchor="top",
            font=dict(
                color="#172033",
                size=18,
            ),
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.08,
            xanchor="left",
            x=0.01,
            title=dict(
                text="",
            ),
            font=dict(
                color="#334155",
                size=13,
            ),
            bgcolor="rgba(255,255,255,0)",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#cbd5e1",
            font=dict(
                color="#172033",
            ),
        ),
    )

    fig.update_xaxes(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        zerolinecolor="#cbd5e1",
        tickfont=dict(
            color="#475569",
        ),
        title_font=dict(
            color="#334155",
        ),
    )

    fig.update_yaxes(
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
        zerolinecolor="#cbd5e1",
        tickfont=dict(
            color="#475569",
        ),
        title_font=dict(
            color="#334155",
        ),
    )

    return fig


def furnace_figure(row, twin):
    if twin["state"] == "STABLE": color = PALETTE["green"]
    elif twin["state"] == "HANGING RISK": color = PALETTE["red"]
    else: color = PALETTE["amber"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[-1.5,-2.3,-3.0,-2.5,-1.5,-1.1,1.1,1.5,2.5,3.0,2.3,1.5,-1.5],
        y=[10,8.3,5.6,3.2,1.2,0,0,1.2,3.2,5.6,8.3,10,10],
        fill="toself", line=dict(color="#475569", width=3), fillcolor=color,
        opacity=.55, hoverinfo="skip", mode="lines"
    ))
    zones=[("THROAT",9.2),("STACK",7.3),("COHESIVE ZONE",5.3),("BOSH",3.2),("HEARTH",1.0)]
    for name,y in zones:
        fig.add_annotation(x=0,y=y,text=name,showarrow=False,font=dict(size=11,color=PALETTE["text"]))
    fig.add_annotation(x=0,y=6.2,text=f"<b>{row['top_gas_temperature_c']:.0f} °C</b>" if pd.notna(row['top_gas_temperature_c']) else "<b>NO DATA</b>",showarrow=False,font=dict(size=15,color=PALETTE["text"]))
    fig.add_annotation(x=0,y=2.0,text=f"<b>HM {row['hot_metal_temperature_c']:.0f} °C</b>",showarrow=False,font=dict(size=15,color=PALETTE["text"]))
    fig.update_layout(xaxis=dict(visible=False,range=[-4,4]),yaxis=dict(visible=False,range=[-0.4,10.5]),showlegend=False)
    return apply_chart_theme(fig, 440)


def gauge(value, title, max_value=100):
    fig=go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text":title},
        gauge={"axis":{"range":[0,max_value], "tickcolor":PALETTE["text"]},"bar":{"color":PALETTE['blue']},
               "steps":[{"range":[0,50],"color":"#f8d7da"},{"range":[50,75],"color":"#fff3cd"},{"range":[75,100],"color":"#d1e7dd"}]}
    ))
    return apply_chart_theme(fig,250)


def correlation_figure(df, columns):
    corr=df[columns].corr()
    fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",zmin=-1,zmax=1)
    return apply_chart_theme(fig,500)

# ============================================================
# SIDEBAR AND DATA SOURCE
# ============================================================

st.sidebar.markdown("## 🏭 BF1 Digital Twin")
st.sidebar.caption(f"Version {APP_VERSION}")
source_mode = st.sidebar.radio("Data source", ["Included synthetic historian", "Upload historian CSV"])

if source_mode == "Upload historian CSV":
    uploaded = st.sidebar.file_uploader("Upload plant-approved CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV or select the included synthetic historian.")
        st.stop()
    source = uploaded
else:
    source = DEFAULT_FILE

try:
    df = load_csv(source)
except Exception as exc:
    st.error(f"Data validation failed: {exc}")
    st.stop()

start_time = df["timestamp"].min().to_pydatetime()
end_time = df["timestamp"].max().to_pydatetime()

st.sidebar.success(f"{len(df):,} historian records loaded")
if "replay_cursor" not in st.session_state:
    st.session_state.replay_cursor = start_time
if st.session_state.replay_cursor < start_time or st.session_state.replay_cursor > end_time:
    st.session_state.replay_cursor = start_time
replay_time = st.sidebar.slider(
    "Historical replay position", min_value=start_time, max_value=end_time,
    step=timedelta(minutes=5), key="replay_cursor"
)
window_hours = st.sidebar.slider("Analysis window (hours)", 1, 48, 8)
auto_replay = st.sidebar.checkbox("Auto replay", value=False)
replay_step = st.sidebar.select_slider("Auto-replay step", options=[1,5,10,15,30,60], value=10, format_func=lambda x:f"{x} min")

closest_index = int((df["timestamp"] - pd.Timestamp(replay_time)).abs().idxmin())
row = df.loc[closest_index]
previous = df.loc[max(0, closest_index-1)] if closest_index > 0 else None
window = df[(df["timestamp"] <= row["timestamp"]) & (df["timestamp"] >= row["timestamp"] - pd.Timedelta(hours=window_hours))].copy()
twin = evaluate_twin(row, previous)
forecast_30, uncertainty_30, confidence_30 = forecast_temperature(window, 30)
forecast_si = forecast_silicon(window, forecast_30)

st.sidebar.markdown("---")
st.sidebar.markdown("**Replay timestamp**")
st.sidebar.code(str(row["timestamp"]))
st.sidebar.caption("No PLC/DCS/SCADA connection. No write-back capability.")

# ============================================================
# HEADER AND SUMMARY
# ============================================================


# Summary metrics in a single horizontal row
st.title("Blast Furnace Digital Twin")

header1, header2, header3, header4, header5, header6 = st.columns(    [0.72, 1.02, 1.15, 1.05, 1.10, 0.85],    gap="small",)
with header1:
    st.markdown(
        f'<div class="state-card {twin["css"]}">{twin["state"]}</div>',
        unsafe_allow_html=True,
    )

with header2:
    st.metric(
        "Stability",
        f"{twin['score']}/100",
    )

with header3:
    st.metric(
        "HM Temp.",
        f"{row['hot_metal_temperature_c']:.1f} °C",
    )

with header4:
    st.metric(
        "HM Silicon",
        f"{row['hot_metal_silicon_pct']:.3f} %",
    )

with header5:
    st.metric(
        "Gas Util.",
        f"{row['gas_utilisation_pct']:.1f} %",
    )

with header6:
    st.metric(
        "Quality",
        str(row["quality"]),
    )

# ============================================================
# APPLICATION TABS
# ============================================================

(
    command_tab, furnace_tab, trends_tab, burden_tab, gas_tab,
    cooling_tab, prediction_tab, alarm_tab, scenario_tab,
    quality_tab, analytics_tab, report_tab
) = st.tabs([
    "Command Centre", "Furnace", "Trends", "Burden & Fuel", "Gas & Pressure",
    "Cooling", "Predictions", "Alarms", "Scenario Lab", "Data Quality",
    "Analytics", "Shift Report"
])

with command_tab:
    left, middle, right = st.columns([1.2,1,1])
    with left:
        st.plotly_chart(
    furnace_figure(row, twin),
    use_container_width=True,
    key="command_centre_furnace_chart",
)

    with middle:
        st.plotly_chart(gauge(twin["score"],"Stability score"),use_container_width=True)
        kpi=pd.DataFrame({
            "KPI":["RAFT proxy","Permeability index","CO/CO₂ ratio","Thermal index"],
            "Value":[f"{row['raft_proxy_c']:.0f} °C",f"{row['permeability_index']:.3f}",f"{row['co_co2_ratio']:.3f}",f"{row['thermal_index']:.1f}/100"]
        })
        st.dataframe(kpi,hide_index=True,use_container_width=True)
    with right:
        st.subheader("30-minute advisory outlook")
        if pd.isna(forecast_30):
            st.warning("Insufficient history for forecast")
        else:
            st.metric("Forecast HM temperature",f"{forecast_30:.1f} °C",delta=f"{forecast_30-row['hot_metal_temperature_c']:+.1f} °C")
            st.metric("Uncertainty band",f"±{uncertainty_30:.1f} °C")
            st.metric("Forecast silicon",f"{forecast_si:.3f} %")
            st.caption(f"Confidence: {confidence_30}; transparent trend/process proxy, not a trained plant model.")
        st.subheader("Active advisories")
        if twin["alarms"]:
            for alarm in twin["alarms"][:5]:
                if alarm["Severity"] in ("CRITICAL","HIGH"): st.error(f"{alarm['Severity']} — {alarm['Message']}")
                else: st.warning(f"{alarm['Severity']} — {alarm['Message']}")
        else:
            st.success("No active prototype advisories")

with furnace_tab:
    a, b = st.columns([1, 1.4])

    with a:
        st.plotly_chart(
            furnace_figure(row, twin),
            use_container_width=True,
            key="furnace_tab_furnace_chart",
        )

    with b:
        st.subheader("Furnace state model")

        state_table = pd.DataFrame(
            [
                [
                    "Throat / top gas",
                    row["top_gas_temperature_c"],
                    "°C",
                    "Top-gas thermal condition",
                ],
                [
                    "Stack gas utilization",
                    row["gas_utilisation_pct"],
                    "%",
                    "CO₂/(CO+CO₂) proxy",
                ],
                [
                    "Gas permeability",
                    row["permeability_index"],
                    "index",
                    "ΔP / blast pressure",
                ],
                [
                    "Raceway thermal proxy",
                    row["raft_proxy_c"],
                    "°C",
                    "Transparent demonstration relationship",
                ],
                [
                    "Hearth thermal state",
                    row["hot_metal_temperature_c"],
                    "°C",
                    "Hot-metal temperature",
                ],
                [
                    "Hot-metal chemistry",
                    row["hot_metal_silicon_pct"],
                    "% Si",
                    "Synthetic laboratory value",
                ],
            ],
            columns=["Zone / state", "Value", "Unit", "Basis"],
        )

        st.dataframe(
            state_table,
            hide_index=True,
            use_container_width=True,
        )
    with b:
        st.subheader("Furnace state model")
        state_table=pd.DataFrame([
            ["Throat / top gas",row["top_gas_temperature_c"],"°C","Top-gas thermal condition"],
            ["Stack gas utilization",row["gas_utilisation_pct"],"%","CO₂/(CO+CO₂) proxy"],
            ["Gas permeability",row["permeability_index"],"index","ΔP / blast pressure"],
            ["Raceway thermal proxy",row["raft_proxy_c"],"°C","Transparent demonstration relationship"],
            ["Hearth thermal state",row["hot_metal_temperature_c"],"°C","Hot-metal temperature"],
            ["Hot-metal chemistry",row["hot_metal_silicon_pct"],"% Si","Synthetic laboratory value"],
        ],columns=["Zone / state","Value","Unit","Basis"])
        st.dataframe(state_table,hide_index=True,use_container_width=True)

with trends_tab:
    selected_tags=st.multiselect("Select trend tags",options=[c for c in REQUIRED_COLUMNS if c not in ("timestamp","quality")],default=["hot_metal_temperature_c","hot_blast_temperature_c","differential_pressure_bar","top_gas_temperature_c"])
    if selected_tags:
        normalized=st.checkbox("Normalize selected signals",value=False)
        chart_df=window[["timestamp"]+selected_tags].copy()
        if normalized:
            for c in selected_tags:
                minv,maxv=chart_df[c].min(),chart_df[c].max()
                chart_df[c]=(chart_df[c]-minv)/(maxv-minv) if pd.notna(maxv) and maxv!=minv else 0
        melted=chart_df.melt("timestamp",var_name="Tag",value_name="Value")
        fig=px.line(melted,x="timestamp",y="Value",color="Tag",title=f"{window_hours}-hour historian trends")
        st.plotly_chart(apply_chart_theme(fig,500),use_container_width=True)
    quick1,quick2=st.columns(2)
    with quick1:
        fig=px.line(window,x="timestamp",y=["hot_metal_temperature_c","hot_blast_temperature_c"],title="Thermal trend")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
    with quick2:
        fig=px.line(window,x="timestamp",y=["blast_pressure_bar","differential_pressure_bar"],title="Pressure trend")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)

with burden_tab:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("PCI",f"{row['pci_rate_kg_thm']:.1f} kg/tHM")
    c2.metric("Stockline",f"{row['stockline_m']:.2f} m")
    c3.metric("Specific blast proxy",f"{row['specific_blast_proxy']:.1f}")
    c4.metric("Fuel efficiency proxy",f"{max(0,100-(row['pci_rate_kg_thm']-150)*0.5):.1f}/100")
    fig=make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(x=window.timestamp,y=window.pci_rate_kg_thm,name="PCI kg/tHM"),secondary_y=False)
    fig.add_trace(go.Scatter(x=window.timestamp,y=window.stockline_m,name="Stockline m"),secondary_y=True)
    fig.update_yaxes(title_text="PCI kg/tHM",secondary_y=False)
    fig.update_yaxes(title_text="Stockline m",secondary_y=True)
    st.plotly_chart(apply_chart_theme(fig,430),use_container_width=True)
    st.caption("Burden batch weights, coke rate, ore/coke ratio, size distribution, chemistry and charging matrix are not present in the current synthetic dataset and should be added when approved data becomes available.")

with gas_tab:
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Blast pressure",f"{row['blast_pressure_bar']:.2f} bar")
    c2.metric("Differential pressure",f"{row['differential_pressure_bar']:.2f} bar")
    c3.metric("Permeability index",f"{row['permeability_index']:.3f}")
    c4.metric("Gas utilisation",f"{row['gas_utilisation_pct']:.1f} %")
    a,b=st.columns(2)
    with a:
        fig=px.line(window,x="timestamp",y=["top_gas_co_pct","top_gas_co2_pct"],title="Top-gas composition")
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
    with b:
        fig=px.scatter(window,x="differential_pressure_bar",y="gas_utilisation_pct",color="hot_metal_temperature_c",title="Pressure loss vs gas utilisation",color_continuous_scale="Turbo")
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)

with cooling_tab:
    c1,c2,c3=st.columns(3)
    c1.metric("Cooling-water ΔT",f"{row['cooling_water_delta_t_c']:.2f} °C")
    c2.metric("Cooling risk proxy",f"{min(100,max(0,(row['cooling_water_delta_t_c']-6)*20)):.0f}/100")
    c3.metric("Cooling data status","AVAILABLE" if pd.notna(row['cooling_water_delta_t_c']) else "UNAVAILABLE")
    fig=px.line(window,x="timestamp",y="cooling_water_delta_t_c",title="Cooling-water temperature rise")
    fig.add_hline(y=LIMITS["cooling_water_delta_t_c"]["high"],line_dash="dash",line_color=PALETTE["red"],annotation_text="Prototype high band")
    st.plotly_chart(apply_chart_theme(fig,420),use_container_width=True)
    st.info("A real cooling twin should include circuit-level inlet/outlet temperature, flow, pressure, heat flux, stave/shell temperatures, leak detection and sensor redundancy.")

with prediction_tab:
    st.subheader("Advisory predictions")
    horizons=[15,30,60]
    forecasts=[]
    for horizon in horizons:
        f,u,c=forecast_temperature(window,horizon)
        forecasts.append({"Horizon":f"{horizon} min","HM temperature forecast °C":f,"Uncertainty ±°C":u,"Confidence":c})
    st.dataframe(pd.DataFrame(forecasts),hide_index=True,use_container_width=True)
    history_plot=window.tail(180).copy()
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=history_plot.timestamp,y=history_plot.hot_metal_temperature_c,name="Measured",mode="lines"))
    if pd.notna(forecast_30):
        future_time=row["timestamp"]+pd.Timedelta(minutes=30)
        fig.add_trace(go.Scatter(x=[row["timestamp"],future_time],y=[row["hot_metal_temperature_c"],forecast_30],name="30-min forecast",mode="lines+markers",line=dict(dash="dash")))
        fig.add_trace(go.Scatter(x=[future_time,future_time],y=[forecast_30-uncertainty_30,forecast_30+uncertainty_30],name="Uncertainty",mode="lines+markers"))
    st.plotly_chart(apply_chart_theme(fig,430),use_container_width=True)

with alarm_tab:
    st.subheader("Alarm and advisory console")
    severity_filter=st.multiselect("Severity",["CRITICAL","HIGH","MEDIUM"],default=["CRITICAL","HIGH","MEDIUM"])
    alarm_df=pd.DataFrame(twin["alarms"])
    if alarm_df.empty:
        st.success("No active alarms at the selected replay timestamp")
    else:
        alarm_df=alarm_df[alarm_df["Severity"].isin(severity_filter)]
        st.dataframe(alarm_df,hide_index=True,use_container_width=True)
    st.subheader("Recent event scan")
    event_rows=[]
    scan=window.tail(min(len(window),500))
    for index,r in scan.iterrows():
        prev=df.loc[index-1] if index>0 else None
        result=evaluate_twin(r,prev)
        for alarm in result["alarms"]:
            event_rows.append({"Timestamp":r["timestamp"],**alarm})
    events=pd.DataFrame(event_rows)
    if events.empty: st.info("No events detected in the selected window")
    else:
        st.dataframe(events.sort_values("Timestamp",ascending=False).head(250),hide_index=True,use_container_width=True)
        st.download_button("Download event log",events.to_csv(index=False).encode(),"bf_event_log.csv","text/csv")

with scenario_tab:
    st.subheader("Scenario laboratory")
    st.caption("Changes are hypothetical and never alter the source historian data.")
    s1,s2,s3,s4,s5=st.columns(5)
    with s1: hbt_delta=st.slider("HBT Δ °C",-150,150,0,5)
    with s2: pci_delta=st.slider("PCI Δ kg/tHM",-40,40,0,2)
    with s3: oxygen_delta=st.slider("O₂ Δ %",-2.0,2.0,0.0,0.1)
    with s4: blast_delta=st.slider("Blast flow Δ",-500,500,0,25)
    with s5: moisture_proxy=st.slider("Moisture proxy Δ",-10.0,10.0,0.0,0.5)
    scenario=scenario_engine(row,hbt_delta,pci_delta,oxygen_delta,blast_delta,moisture_proxy)
    r1,r2,r3,r4,r5=st.columns(5)
    r1.metric("HM temperature",f"{scenario['hm_temperature']:.1f} °C",f"{scenario['hm_delta']:+.1f} °C")
    r2.metric("RAFT proxy",f"{scenario['raft']:.0f} °C",f"{scenario['raft']-row['raft_proxy_c']:+.0f} °C")
    r3.metric("Fuel-rate direction",f"{scenario['fuel_delta']:+.1f} kg/tHM")
    r4.metric("Productivity direction",f"{scenario['productivity_delta_pct']:+.1f} %")
    r5.metric("Gas utilisation",f"{scenario['gas_utilisation']:.1f} %")
    radar_categories=["Thermal","Gas utilization","Permeability","Cooling","Stability"]
    baseline=[row['thermal_index'],row['gas_utilisation_pct']*2,min(100,100-row['permeability_index']*100),max(0,100-row['cooling_water_delta_t_c']*5),twin['score']]
    changed=[min(100,max(0,baseline[0]+scenario['hm_delta'])),min(100,max(0,scenario['gas_utilisation']*2)),baseline[2],baseline[3],max(0,min(100,twin['score']-abs(scenario['hm_delta'])*0.8))]
    fig=go.Figure()
    fig.add_trace(go.Scatterpolar(r=baseline+[baseline[0]],theta=radar_categories+[radar_categories[0]],fill="toself",name="Baseline"))
    fig.add_trace(go.Scatterpolar(r=changed+[changed[0]],theta=radar_categories+[radar_categories[0]],fill="toself",name="Scenario"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100])),showlegend=True)
    st.plotly_chart(apply_chart_theme(fig,430),use_container_width=True)

with quality_tab:
    report=data_quality_report(df)
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Rows",f"{len(df):,}")
    q2.metric("Non-GOOD rows",f"{int((df['quality']!='GOOD').sum()):,}")
    q3.metric("Missing values",f"{int(df[REQUIRED_COLUMNS].isna().sum().sum()):,}")
    expected_seconds=df["timestamp"].diff().dt.total_seconds().median()
    q4.metric("Median sampling",f"{expected_seconds:.0f} s")
    st.dataframe(report,hide_index=True,use_container_width=True)
    fig=px.bar(report,x="Tag",y=["Missing","Outside prototype band","Flat-line points"],barmode="group",title="Data-quality findings")
    st.plotly_chart(apply_chart_theme(fig,430),use_container_width=True)
    st.download_button("Download data-quality report",report.to_csv(index=False).encode(),"bf_data_quality_report.csv","text/csv")

with analytics_tab:
    analysis_columns=["hot_blast_temperature_c","hot_blast_flow_nm3_min","blast_pressure_bar","differential_pressure_bar","pci_rate_kg_thm","oxygen_enrichment_pct","top_gas_temperature_c","top_gas_co_pct","top_gas_co2_pct","cooling_water_delta_t_c","hot_metal_temperature_c","hot_metal_silicon_pct"]
    st.plotly_chart(correlation_figure(window,analysis_columns),use_container_width=True)
    a,b=st.columns(2)
    with a:
        fig=px.scatter(window,x="hot_blast_temperature_c",y="hot_metal_temperature_c",color="differential_pressure_bar",trendline=None,title="Hot blast vs hot metal temperature")
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
    with b:
        fig=px.scatter(window,x="gas_utilisation_pct",y="hot_metal_silicon_pct",color="hot_metal_temperature_c",title="Gas utilisation vs silicon")
        st.plotly_chart(apply_chart_theme(fig),use_container_width=True)
    st.caption("Association does not establish causation. Blast-furnace signals require time alignment and process-lag analysis before modelling.")

with report_tab:
    st.subheader("Shift / engineering report")
    report_hours=st.selectbox("Report period",[4,8,12,24],index=1)
    report_window=df[(df.timestamp<=row.timestamp)&(df.timestamp>=row.timestamp-pd.Timedelta(hours=report_hours))]
    summary=pd.DataFrame({
        "Metric":["Start","End","Records","GOOD quality %","Average HM temperature °C","HM temperature min °C","HM temperature max °C","Average silicon %","Average gas utilisation %","Average differential pressure bar","Maximum differential pressure bar"],
        "Value":[report_window.timestamp.min(),report_window.timestamp.max(),len(report_window),round(100*(report_window.quality=='GOOD').mean(),2),round(report_window.hot_metal_temperature_c.mean(),2),round(report_window.hot_metal_temperature_c.min(),2),round(report_window.hot_metal_temperature_c.max(),2),round(report_window.hot_metal_silicon_pct.mean(),3),round(report_window.gas_utilisation_pct.mean(),2),round(report_window.differential_pressure_bar.mean(),3),round(report_window.differential_pressure_bar.max(),3)]
    })
    st.dataframe(summary,hide_index=True,use_container_width=True)
    report_text=(
        f"BF1 OFFLINE ADVISORY REPORT\n"
        f"Period: {report_window.timestamp.min()} to {report_window.timestamp.max()}\n"
        f"Current state: {twin['state']}\n"
        f"Stability score: {twin['score']}/100\n"
        f"Current HM temperature: {row['hot_metal_temperature_c']:.1f} C\n"
        f"Current HM silicon: {row['hot_metal_silicon_pct']:.3f} %\n"
        f"Current gas utilisation proxy: {row['gas_utilisation_pct']:.1f} %\n"
        f"Active advisory count: {len(twin['alarms'])}\n"
        f"\nDISCLAIMER: Synthetic/offline advisory prototype. Not valid for plant control or operating decisions.\n"
    )
    st.download_button("Download text report",report_text.encode(),"bf_shift_report.txt","text/plain")
    st.download_button("Download report-period data",report_window.to_csv(index=False).encode(),"bf_shift_data.csv","text/csv")

st.markdown("---")
st.caption("Synthetic-data engineering prototype. Models and limits are not plant-validated. No control or write-back capability.")

if auto_replay:
    next_time = pd.Timestamp(replay_time) + pd.Timedelta(minutes=replay_step)
    if next_time > df["timestamp"].max():
        next_time = df["timestamp"].min()
    time.sleep(1.5)
    st.session_state.replay_cursor = next_time.to_pydatetime()
    st.rerun()
