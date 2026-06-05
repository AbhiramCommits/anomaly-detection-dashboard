#!/usr/bin/env bash
set -e

echo "=== Anomaly Detection Dashboard — EC2 Setup ==="

# ---------- system updates ----------
echo "[1/5] Updating system packages..."
sudo apt-get update -y && sudo apt-get upgrade -y

# ---------- install python ----------
echo "[2/5] Installing Python 3 and pip..."
sudo apt-get install -y python3 python3-pip python3-venv

# ---------- install project dependencies ----------
echo "[3/5] Installing project dependencies..."
cd "$(dirname "$0")/.."
pip3 install -r requirements.txt

# ---------- generate data if not present ----------
echo "[4/5] Generating sample data..."
if [ ! -f data/process_stream.csv ]; then
    python3 src/generate_data.py
fi

# ---------- start streamlit ----------
echo "[5/5] Starting Streamlit dashboard on port 8501..."
nohup streamlit run app/dashboard.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &

echo ""
echo "=== Setup complete ==="
echo "Dashboard running at http://$(curl -s http://checkip.amazonaws.com):8501"
echo "Logs: tail -f streamlit.log"
