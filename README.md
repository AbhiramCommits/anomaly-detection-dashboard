# Anomaly Detection Dashboard

[![Lint](https://github.com/your-org/anomaly-detection-dashboard/actions/workflows/lint.yml/badge.svg)](https://github.com/your-org/anomaly-detection-dashboard/actions/workflows/lint.yml)

A machine learning pipeline for real-time process sensor anomaly detection, combining Isolation Forest and LSTM Autoencoder models with a Streamlit dashboard for interactive monitoring.

## Overview

This project simulates a **Digital Twin** of an industrial process environment. Four sensor streams — temperature, pressure, flow rate, and vibration — are continuously monitored. Anomalies are injected into 3% of the data and detected using two complementary approaches:

- **Isolation Forest** — tree-based unsupervised method that isolates outliers via random partitioning
- **LSTM Autoencoder** — deep learning model that learns normal behavior and flags sequences with high reconstruction error

The flagged data is surfaced in an interactive Streamlit dashboard for operators to inspect and respond to anomalies in near real-time.

## Architecture

```
                  ┌──────────────────────┐
                  │   generate_data.py   │
                  │  Synthetic sensors + │
                  │   injected anomalies │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  process_stream.csv  │
                  │   (50,000 rows × 5)  │
                  └──────────┬───────────┘
                             │
              ┌──────────────┼──────────────┐
              │                             │
   ┌──────────▼──────────┐    ┌─────────────▼────────────┐
   │  isolation_forest   │    │   lstm_autoencoder.py    │
   │  sklearn / IF       │    │   TensorFlow / Keras     │
   │  contamination=0.03 │    │   LSTM(64)→LSTM(32)→     │
   │                     │    │   RepeatVector→          │
   │  anomaly_score      │    │   LSTM(32)→LSTM(64)→Dense│
   │  anomaly_flag       │    │                          │
   └──────────┬──────────┘    └─────────────┬────────────┘
              │                             │
              └──────────────┬──────────────┘
                             │
                  ┌──────────▼───────────┐
                  │  flagged_stream.csv  │
                  │  + anomaly_score     │
                  │  + anomaly_flag      │
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │   app/dashboard.py   │
                  │   Streamlit UI       │
                  │                      │
                  │  · Sensor line chart │
                  │  · Anomaly overlay   │
                  │  · Metrics row       │
                  │  · Anomaly table     │
                  └──────────────────────┘
```

## Model Comparison

| Model              | Precision | Recall | F1 Score |
|--------------------|-----------|--------|----------|
| Isolation Forest   | 1.0000    | 1.0000 | 1.0000   |
| LSTM Autoencoder   | 0.7593    | 0.7593 | 0.7593   |

Isolation Forest achieves perfect scores on this dataset because the injected anomalies (2.5–4× multiplier) create trivially separable feature-space outliers.

The LSTM Autoencoder uses a fixed 97th-percentile threshold on reconstruction error, which flags exactly 3% of samples regardless of true anomaly distribution. Its lower F1 reflects the limitation of a percentile-based cutoff rather than a learned decision boundary.

Full metrics saved to [`results/detection_metrics.csv`](results/detection_metrics.csv).

## Folder Structure

```
anomaly-detection-dashboard/
├── app/
│   └── dashboard.py                # Streamlit monitoring dashboard
├── src/
│   ├── generate_data.py            # Synthetic sensor data generator
│   ├── isolation_forest.py         # Isolation Forest training + evaluation
│   └── lstm_autoencoder.py         # LSTM Autoencoder training + evaluation
├── data/
│   ├── process_stream.csv          # Raw sensor readings (50,000 rows)
│   └── flagged_stream.csv          # Anomaly-flagged dataset
├── models/
│   ├── isolation_forest.pkl        # Trained IF model
│   └── lstm_autoencoder.h5         # Trained LSTM autoencoder
├── results/
│   └── detection_metrics.csv       # Model evaluation metrics
├── deploy/
│   ├── setup.sh                    # Ubuntu EC2 setup script
│   └── README.md                   # AWS deployment guide
├── .github/workflows/
│   └── lint.yml                    # flake8 CI on push to main
└── requirements.txt                # Python dependencies
```

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-org/anomaly-detection-dashboard.git
cd anomaly-detection-dashboard

# 2. (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data
python src/generate_data.py

# 5. Train models
python src/isolation_forest.py
python src/lstm_autoencoder.py

# 6. Launch dashboard
streamlit run app/dashboard.py
```

Open `http://localhost:8501` in your browser.

## EC2 Deployment

See [`deploy/README.md`](deploy/README.md) for detailed instructions on launching an Ubuntu EC2 instance, configuring the security group, and running the one-command setup script:

```bash
./deploy/setup.sh
```

The dashboard will be available at `http://<public-ip>:8501`.

## Digital Twin Monitoring

This project models a **Digital Twin** — a virtual representation of a physical industrial process — for real-time equipment health monitoring:

- **Sensor Simulation:** Four sensor types (temperature, pressure, flow rate, vibration) are generated from Gaussian distributions mimicking normal operating ranges of a pump, compressor, or rotating machine.
- **Fault Injection:** 3% of data points are randomly amplified by a factor of 2.5–4×, simulating real-world failure modes such as bearing wear, seal leakage, cavitation, or overpressure events.
- **Dual-Model Detection:** The Isolation Forest catches extreme outliers in the feature space, while the LSTM Autoencoder captures temporal anomalies that deviate from learned normal sequence patterns — together providing a defense-in-depth monitoring strategy.
- **Operator Dashboard:** The Streamlit UI mirrors a plant operator's HMI/SCADA screen, with time-window filtering, visual anomaly overlays, live metrics, and a drill-down table of flagged events.

In a production setting, the CSV source would be replaced with a live stream from MQTT, OPC-UA, or AWS Kinesis, and the models would run inference continuously on incoming windows of sensor data.
