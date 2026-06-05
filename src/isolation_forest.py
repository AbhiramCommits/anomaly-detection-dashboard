import pandas as pd
import numpy as np
import os
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

np.random.seed(42)

ROWS = 50_000
ANOMALY_PCT = 0.03
SENSOR_COLS = ["temp_sensor", "pressure_sensor", "flow_rate", "vibration_sensor"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
INPUT_FILE = os.path.join(DATA_DIR, "process_stream.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "flagged_stream.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "isolation_forest.pkl")

# -- reconstruct ground truth (same RNG sequence as generate_data.py) --
_ = np.random.normal(size=ROWS)  # temp
_ = np.random.normal(size=ROWS)  # pressure
_ = np.random.normal(size=ROWS)  # flow_rate
_ = np.random.normal(size=ROWS)  # vibration
n_anomalies = int(ROWS * ANOMALY_PCT)
anomaly_indices = np.random.choice(ROWS, size=n_anomalies, replace=False)
# -------------------------------------------------------------------

df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])

X = df[SENSOR_COLS].values

model = IsolationForest(contamination=ANOMALY_PCT, random_state=42)
model.fit(X)

df["anomaly_score"] = -model.decision_function(X)
df["anomaly_flag"] = (model.predict(X) == -1).astype(int)

y_true = np.zeros(ROWS, dtype=int)
y_true[anomaly_indices] = 1
y_pred = df["anomaly_flag"].values

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"Flagged df saved to: {OUTPUT_FILE}")
print(f"Model saved to:      {MODEL_FILE}")
