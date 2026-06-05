# EC2 Deployment Guide

This guide walks through deploying the Anomaly Detection Dashboard on a fresh Ubuntu EC2 instance.

## Prerequisites

- AWS account with EC2 access
- Git installed locally
- SSH key pair created in AWS (`.pem` file downloaded)

## Step 1 — Launch EC2 Instance

1. Go to **EC2 Dashboard → Launch Instance**
2. **Name:** `anomaly-dashboard`
3. **AMI:** Ubuntu Server 22.04 LTS (or 24.04)
4. **Instance type:** `t2.medium` or larger (TensorFlow needs >= 4 GB RAM)
5. **Key pair:** Select your existing key or create a new one
6. **Network settings:**
   - Allow SSH (port 22) from your IP
   - Add a **custom TCP rule** for port **8501** (Streamlit) with source `0.0.0.0/0`
7. **Storage:** 20 GB gp2 (minimum)
8. Click **Launch instance**

## Step 2 — Connect to the Instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<public-ipv4-dns>
```

## Step 3 — Clone the Repository

```bash
git clone https://github.com/your-org/anomaly-detection-dashboard.git
cd anomaly-detection-dashboard
```

## Step 4 — Run Setup Script

```bash
chmod +x deploy/setup.sh
./deploy/setup.sh
```

This script will:
- Update system packages
- Install Python 3, pip, and venv
- Install all Python dependencies from `requirements.txt`
- Generate sample data if not already present
- Start the Streamlit dashboard on port 8501 via nohup

## Step 5 — Access the Dashboard

Open your browser and navigate to:

```
http://<public-ipv4>:8501
```

Find your instance's public IP in the EC2 console or run:

```bash
curl -s http://checkip.amazonaws.com
```

## Step 6 — Verify & Manage

```bash
# Check if Streamlit is running
ps aux | grep streamlit

# View logs
tail -f streamlit.log

# Stop the dashboard
pkill -f "streamlit run"

# Restart the dashboard
nohup streamlit run app/dashboard.py --server.port 8501 --server.headless true > streamlit.log 2>&1 &
```

## Architecture

```
anomaly-detection-dashboard/
├── app/dashboard.py          # Streamlit UI
├── src/
│   ├── generate_data.py      # Synthetic data generator
│   ├── isolation_forest.py   # Isolation Forest model
│   └── lstm_autoencoder.py   # LSTM Autoencoder model
├── data/
│   ├── process_stream.csv    # Raw sensor data
│   └── flagged_stream.csv    # Anomaly-flagged dataset
├── models/
│   ├── isolation_forest.pkl
│   └── lstm_autoencoder.h5
└── deploy/
    ├── setup.sh
    └── README.md
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8501 unreachable | Verify security group inbound rule for TCP/8501 |
| `pip3: command not found` | Run `sudo apt-get install -y python3-pip` |
| Out of memory (OOM) | Use `t2.medium` or larger; TensorFlow needs ~4 GB |
| Streamlit not starting | Check `tail -f streamlit.log` for errors |
