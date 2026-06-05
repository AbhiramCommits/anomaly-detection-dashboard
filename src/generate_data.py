import pandas as pd
import numpy as np
import os

np.random.seed(42)

ROWS = 50_000
ANOMALY_PCT = 0.03
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "process_stream.csv")

timestamps = pd.date_range(start="2025-01-01", periods=ROWS, freq="1min")

temp = np.random.normal(loc=75.0, scale=5.0, size=ROWS)
pressure = np.random.normal(loc=30.0, scale=3.0, size=ROWS)
flow_rate = np.random.normal(loc=100.0, scale=8.0, size=ROWS)
vibration = np.abs(np.random.normal(loc=0.5, scale=0.1, size=ROWS))

n_anomalies = int(ROWS * ANOMALY_PCT)
anomaly_indices = np.random.choice(ROWS, size=n_anomalies, replace=False)
anomaly_factors = np.random.uniform(2.5, 4.0, size=n_anomalies)

for i, idx in enumerate(anomaly_indices):
    factor = anomaly_factors[i]
    temp[idx] *= factor
    pressure[idx] *= factor
    flow_rate[idx] *= factor
    vibration[idx] *= factor

df = pd.DataFrame({
    "timestamp": timestamps,
    "temp_sensor": temp,
    "pressure_sensor": pressure,
    "flow_rate": flow_rate,
    "vibration_sensor": vibration,
})

os.makedirs(OUTPUT_DIR, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Total rows: {len(df)}")
print(f"Anomaly count: {n_anomalies} ({ANOMALY_PCT*100:.0f}%)")
print(f"Saved to: {OUTPUT_FILE}")
