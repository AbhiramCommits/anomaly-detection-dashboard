import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

np.random.seed(42)
tf.random.set_seed(42)

ROWS = 50_000
ANOMALY_PCT = 0.03
SEQ_LEN = 30
SENSOR_COLS = ["temp_sensor", "pressure_sensor", "flow_rate", "vibration_sensor"]

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
INPUT_FILE = os.path.join(DATA_DIR, "process_stream.csv")
FLAGGED_FILE = os.path.join(DATA_DIR, "flagged_stream.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "lstm_autoencoder.h5")

# ---------- 1. Load & normalize ----------
df = pd.read_csv(INPUT_FILE, parse_dates=["timestamp"])
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[SENSOR_COLS].values)  # (50000, 4)

# ---------- 2. Reshape into sequences ----------
n_samples = len(scaled)
sequences = np.array([scaled[i: i + SEQ_LEN] for i in range(n_samples - SEQ_LEN + 1)])
# shape: (49971, 30, 4)

# ---------- 3. Train/val split ----------
X_train, X_val = train_test_split(sequences, test_size=0.1, random_state=42, shuffle=False)

# ---------- 4. Build LSTM autoencoder ----------
input_layer = Input(shape=(SEQ_LEN, 4))
# Encoder
encoded = LSTM(64, return_sequences=True, activation="relu")(input_layer)
encoded = LSTM(32, return_sequences=False, activation="relu")(encoded)
# Decoder
decoded = RepeatVector(SEQ_LEN)(encoded)
decoded = LSTM(32, return_sequences=True, activation="relu")(decoded)
decoded = LSTM(64, return_sequences=True, activation="relu")(decoded)
decoded = TimeDistributed(Dense(4))(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# ---------- 5. Train ----------
history = autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=256,
    validation_data=(X_val, X_val),
    verbose=1,
)

# ---------- 6. Reconstruction error per sample ----------
all_recon = autoencoder.predict(sequences, batch_size=512, verbose=0)  # (49971, 30, 4)

# Map sequence-level squared errors back to per-row errors
error_sum = np.zeros(n_samples)
error_count = np.zeros(n_samples)

for i in range(len(sequences)):
    se = np.square(all_recon[i] - sequences[i]).mean(axis=1)  # (30,) mean over 4 features
    for t, err in enumerate(se):
        row_idx = i + t
        error_sum[row_idx] += err
        error_count[row_idx] += 1

per_sample_re = error_sum / np.maximum(error_count, 1)

# ---------- 7. Flag anomalies (97th percentile) ----------
threshold = np.percentile(per_sample_re, 97)
lstm_pred = (per_sample_re > threshold).astype(int)

# ---------- 8. Ground truth & evaluation ----------
# Reconstruct ground truth (same RNG as generate_data.py)
np.random.seed(42)
_ = np.random.normal(size=ROWS)
_ = np.random.normal(size=ROWS)
_ = np.random.normal(size=ROWS)
_ = np.random.normal(size=ROWS)
n_anomalies = int(ROWS * ANOMALY_PCT)
anomaly_indices = np.random.choice(ROWS, size=n_anomalies, replace=False)

y_true = np.zeros(ROWS, dtype=int)
y_true[anomaly_indices] = 1

# Isolation Forest results from saved file
df_if = pd.read_csv(FLAGGED_FILE)
if_pred = df_if["anomaly_flag"].values

# ---------- 9. Print metrics ----------


def print_metrics(name, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f = f1_score(y_true, y_pred)
    print(f"{name:>22s}  | Precision: {p:.4f}  | Recall: {r:.4f}  | F1: {f:.4f}")


print("\nEvaluation Results")
print("-" * 58)
print_metrics("Isolation Forest", if_pred)
print_metrics("LSTM Autoencoder", lstm_pred)

# ---------- 10. Save model ----------
os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
autoencoder.save(MODEL_FILE)
print(f"\nModel saved to: {MODEL_FILE}")
