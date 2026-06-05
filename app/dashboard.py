import streamlit as st
import pandas as pd
import os
import altair as alt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INPUT_FILE = os.path.join(DATA_DIR, "flagged_stream.csv")

SENSOR_COLS = ["temp_sensor", "pressure_sensor", "flow_rate", "vibration_sensor"]
WINDOW_OPTIONS = {"Last 500 rows": 500, "Last 1,000 rows": 1000, "Last 5,000 rows": 5000}

st.set_page_config(page_title="Anomaly Detection Dashboard", layout="wide")
st.title("Anomaly Detection Dashboard")

# ---------- Load data ----------
@st.cache_data
def load_data():
    return pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

df = load_data()

# ---------- Sidebar ----------
st.sidebar.header("Controls")
sensor = st.sidebar.selectbox("Sensor Column", SENSOR_COLS)
window_label = st.sidebar.selectbox("Time Window", list(WINDOW_OPTIONS.keys()))
n_rows = WINDOW_OPTIONS[window_label]

# ---------- Filter ----------
df_window = df.tail(n_rows)

# ---------- Line chart ----------
df_plot = df_window[["timestamp", sensor, "anomaly_flag"]].copy()
df_plot["color"] = df_plot["anomaly_flag"].map({0: "normal", 1: "anomaly"})

base = alt.Chart(df_plot).encode(x=alt.X("timestamp:T", title="Timestamp"))

line = base.mark_line(color="steelblue").encode(
    y=alt.Y(f"{sensor}:Q", title=sensor),
)

anomaly_points = base.transform_filter(alt.datum.anomaly_flag == 1).mark_circle(
    color="red", size=40
).encode(y=alt.Y(f"{sensor}:Q"))

chart = (line + anomaly_points).properties(height=400).interactive()
st.altair_chart(chart, use_container_width=True)

# ---------- Metrics row ----------
total_anomalies = int(df_window["anomaly_flag"].sum())
anomaly_rate = total_anomalies / len(df_window) * 100
anomalous_rows = df_window[df_window["anomaly_flag"] == 1]
last_anomaly_ts = anomalous_rows["timestamp"].max()
last_anomaly_str = last_anomaly_ts.strftime("%Y-%m-%d %H:%M") if pd.notna(last_anomaly_ts) else "N/A"

c1, c2, c3 = st.columns(3)
c1.metric("Total Anomalies", total_anomalies)
c2.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
c3.metric("Last Anomaly", last_anomaly_str)

# ---------- Anomalous rows table ----------
st.subheader("Anomalous Rows")
st.dataframe(
    anomalous_rows.sort_values("timestamp", ascending=False),
    use_container_width=True,
    hide_index=True,
)
