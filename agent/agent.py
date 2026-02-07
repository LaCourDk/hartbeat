import json
import os
import platform
import socket
import subprocess
import time
from typing import Dict, Optional

import psutil
import requests

SERVER_URL = os.getenv("SERVER_URL", "http://localhost:8080").rstrip("/")
AGENT_ID = os.getenv("AGENT_ID", socket.gethostname())
AGENT_NAME = os.getenv("AGENT_NAME", AGENT_ID)
TAGS = [t.strip() for t in os.getenv("TAGS", "").split(",") if t.strip()]
AGENT_TOKEN = os.getenv("AGENT_TOKEN", "").strip()
INTERVAL = int(os.getenv("INTERVAL", "10"))
HEARTBEAT_EVERY = int(os.getenv("HEARTBEAT_EVERY", "60"))
SERVICES = [s.strip() for s in os.getenv("SERVICES", "").split(",") if s.strip()]


def get_temp() -> Optional[float]:
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        values = []
        for entries in temps.values():
            for entry in entries:
                if entry.current is not None:
                    values.append(entry.current)
        if values:
            return sum(values) / len(values)
    except Exception:
        return None
    return None


def get_gpu() -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            text=True,
        )
        if result.stdout.strip():
            values = [float(v.strip()) for v in result.stdout.splitlines() if v.strip()]
            if values:
                return sum(values) / len(values)
    except Exception:
        return None
    return None


def check_services() -> Optional[Dict[str, str]]:
    if not SERVICES:
        return None
    try:
        running = []
        for proc in psutil.process_iter(attrs=["name"]):
            name = proc.info.get("name") or ""
            if name:
                running.append(name.lower())
        status = {}
        for service in SERVICES:
            found = any(service.lower() in name for name in running)
            status[service] = "ok" if found else "down"
        return status
    except Exception:
        return None


def send_heartbeat() -> None:
    payload = {
        "agent_id": AGENT_ID,
        "name": AGENT_NAME,
        "os": platform.platform(),
        "tags": TAGS,
    }
    services = check_services()
    if services:
        payload["services"] = services
    headers = {"X-Agent-Token": AGENT_TOKEN} if AGENT_TOKEN else None
    requests.post(f"{SERVER_URL}/api/heartbeat", json=payload, timeout=5, headers=headers)


def send_metrics() -> None:
    payload = {
        "agent_id": AGENT_ID,
        "cpu": psutil.cpu_percent(interval=0.2),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "gpu": get_gpu(),
        "temp": get_temp(),
        "net_rx": psutil.net_io_counters().bytes_recv,
        "net_tx": psutil.net_io_counters().bytes_sent,
    }
    services = check_services()
    if services:
        payload["services"] = services
    headers = {"X-Agent-Token": AGENT_TOKEN} if AGENT_TOKEN else None
    requests.post(f"{SERVER_URL}/api/metrics", json=payload, timeout=5, headers=headers)


def main() -> None:
    last_heartbeat = 0.0
    while True:
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_EVERY:
            try:
                send_heartbeat()
                last_heartbeat = now
            except Exception:
                pass
        try:
            send_metrics()
        except Exception:
            pass
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
