# ATMOSAIR v2 — Production Deployment Guide
**Smart India Hackathon 2026 | Problem Statement 26082**

---

## 1. System Requirements
* **Operating System:** Ubuntu 22.04 LTS (recommended) or Windows 11 / Server 2022.
* **CPU:** Minimum 8 vCPUs (16 vCPUs recommended for WRF NetCDF extraction).
* **RAM:** 32 GB RAM minimum.
* **Disk:** 100 GB SSD storage for model weights, cache, and temporary NWP/WRF inputs.
* **Python Runtime:** Python 3.10 to 3.14 with PyTorch, XGBoost, FastAPI, and Uvicorn.

---

## 2. Environment Configuration
Create a `.env` file in the project root based on `.env.example`:

```bash
cp .env.example .env
chmod 600 .env
```

Ensure all required API credentials are populated (`FIRMS_MAP_KEY`, `CAMS_API_KEY`, `SENTINEL_API_CLIENT_ID`, `SENTINEL_API_CLIENT_SECRET`).

---

## 3. Running via Uvicorn Service

To launch the FastAPI backend and static dashboard:
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Access the dashboard at `http://localhost:8000/dashboard/`.

---

## 4. Systemd Service Configuration (Linux Production)

Create `/etc/systemd/system/atmosair.service`:
```ini
[Unit]
Description=ATMOSAIR v2 Air Quality Forecasting Service
After=network.target

[Service]
User=atmosair
WorkingDirectory=/opt/atmosair
ExecStart=/opt/atmosair/venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5
EnvironmentFile=/opt/atmosair/.env

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable atmosair
sudo systemctl start atmosair
```
