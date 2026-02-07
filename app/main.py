from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import httpx
import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from pydantic import BaseModel, Field

DB_PATH = os.getenv("DB_PATH", "/data/hartbeat.db")
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "120"))
OFFLINE_CHECK_INTERVAL = int(os.getenv("OFFLINE_CHECK_INTERVAL", "15"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip() or None

CPU_WARN = float(os.getenv("CPU_WARN", "70"))
CPU_CRIT = float(os.getenv("CPU_CRIT", "90"))
RAM_WARN = float(os.getenv("RAM_WARN", "75"))
RAM_CRIT = float(os.getenv("RAM_CRIT", "90"))
GPU_WARN = float(os.getenv("GPU_WARN", "70"))
GPU_CRIT = float(os.getenv("GPU_CRIT", "90"))
TEMP_WARN = float(os.getenv("TEMP_WARN", "70"))
TEMP_CRIT = float(os.getenv("TEMP_CRIT", "85"))
DISK_WARN = float(os.getenv("DISK_WARN", "80"))
DISK_CRIT = float(os.getenv("DISK_CRIT", "90"))

ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "").strip()
JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))
REQUIRE_AGENT_TOKEN = os.getenv("REQUIRE_AGENT_TOKEN", "1").lower() not in {"0", "false", "no"}

if not JWT_SECRET:
    JWT_SECRET = secrets.token_urlsafe(32)
    print("WARNING: JWT_SECRET not set; using ephemeral secret.")

if not ADMIN_PASSWORD:
    ADMIN_PASSWORD = "admin123"
    print("WARNING: ADMIN_PASSWORD not set; using default 'admin123'.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="Hartbeat Monitor", version="1.0")


class HeartbeatIn(BaseModel):
    agent_id: str = Field(..., min_length=1)
    name: Optional[str] = None
    os: Optional[str] = None
    tags: Optional[List[str]] = None
    services: Optional[Dict[str, str]] = None


class MetricsIn(BaseModel):
    agent_id: str = Field(..., min_length=1)
    ts: Optional[str] = None
    cpu: Optional[float] = None
    ram: Optional[float] = None
    gpu: Optional[float] = None
    temp: Optional[float] = None
    disk: Optional[float] = None
    net_rx: Optional[float] = None
    net_tx: Optional[float] = None
    services: Optional[Dict[str, str]] = None
    extra: Optional[Dict[str, Any]] = None


class LoginIn(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class UserCreate(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)
    role: str = Field("user")


class AgentCreate(BaseModel):
    agent_id: str = Field(..., min_length=1)
    name: Optional[str] = None
    tags: Optional[List[str]] = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(value: Optional[str]) -> datetime:
    if value:
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(username: str, role: str) -> str:
    payload = {
        "sub": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


def require_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    payload = decode_access_token(token)
    conn = get_conn()
    row = conn.execute(
        "SELECT id, username, role FROM users WHERE username = ?",
        (payload.get("sub"),),
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown user")
    return dict(row)


def require_admin(user: Dict[str, Any] = Depends(require_user)) -> Dict[str, Any]:
    if user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")
    return user


def ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    cols = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        conn.commit()


def ensure_admin(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()
    if row and row["count"] == 0:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (ADMIN_USER, hash_password(ADMIN_PASSWORD), "admin", utc_now()),
        )
        conn.commit()


def init_db() -> None:
    conn = get_conn()
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            name TEXT,
            os TEXT,
            tags TEXT,
            last_seen TEXT,
            status TEXT,
            last_health TEXT,
            agent_token TEXT,
            created_at TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT,
            ts TEXT,
            cpu REAL,
            ram REAL,
            gpu REAL,
            temp REAL,
            disk REAL,
            net_rx REAL,
            net_tx REAL,
            services TEXT,
            extra TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_id TEXT,
            ts TEXT,
            level TEXT,
            kind TEXT,
            message TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password_hash TEXT,
            role TEXT,
            created_at TEXT
        );
        """
    )
    ensure_column(conn, "agents", "agent_token", "TEXT")
    ensure_column(conn, "agents", "created_at", "TEXT")
    conn.commit()
    ensure_admin(conn)
    conn.close()


def severity(value: Optional[float], warn: float, crit: float) -> int:
    if value is None:
        return 0
    if value >= crit:
        return 2
    if value >= warn:
        return 1
    return 0


def compute_health(metrics: MetricsIn) -> str:
    level = 0
    level = max(level, severity(metrics.cpu, CPU_WARN, CPU_CRIT))
    level = max(level, severity(metrics.ram, RAM_WARN, RAM_CRIT))
    level = max(level, severity(metrics.gpu, GPU_WARN, GPU_CRIT))
    level = max(level, severity(metrics.temp, TEMP_WARN, TEMP_CRIT))
    level = max(level, severity(metrics.disk, DISK_WARN, DISK_CRIT))
    return "crit" if level == 2 else "warn" if level == 1 else "ok"


def services_status(services: Optional[Dict[str, str]]) -> Optional[str]:
    if not services:
        return None
    for _, status in services.items():
        if status.lower() not in {"ok", "running", "up"}:
            return "down"
    return "ok"


def generate_agent_token() -> str:
    return secrets.token_urlsafe(24)


def require_agent_auth(conn: sqlite3.Connection, agent_id: str, token: Optional[str]) -> None:
    if not REQUIRE_AGENT_TOKEN:
        return
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing agent token")
    row = conn.execute(
        "SELECT agent_token FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if not row:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Unknown agent")
    if not row["agent_token"] or row["agent_token"] != token:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid agent token")


def create_agent_record(
    conn: sqlite3.Connection, agent_id: str, name: Optional[str], tags: Optional[List[str]]
) -> str:
    token = generate_agent_token()
    conn.execute(
        """
        INSERT INTO agents (agent_id, name, tags, status, last_health, last_seen, agent_token, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agent_id,
            name,
            json.dumps(tags or []),
            "offline",
            "ok",
            utc_now(),
            token,
            utc_now(),
        ),
    )
    conn.commit()
    return token


def rotate_agent_token(conn: sqlite3.Connection, agent_id: str) -> str:
    token = generate_agent_token()
    conn.execute(
        "UPDATE agents SET agent_token = ? WHERE agent_id = ?",
        (token, agent_id),
    )
    conn.commit()
    return token


def create_alert(
    conn: sqlite3.Connection,
    agent_id: str,
    level: str,
    kind: str,
    message: str,
) -> Dict[str, Any]:
    payload = {
        "agent_id": agent_id,
        "ts": utc_now(),
        "level": level,
        "kind": kind,
        "message": message,
    }
    conn.execute(
        "INSERT INTO alerts (agent_id, ts, level, kind, message) VALUES (?, ?, ?, ?, ?)",
        (payload["agent_id"], payload["ts"], payload["level"], payload["kind"], payload["message"]),
    )
    conn.commit()
    return payload


def send_webhook(payload: Dict[str, Any]) -> None:
    if not WEBHOOK_URL:
        return
    try:
        httpx.post(WEBHOOK_URL, json=payload, timeout=5.0)
    except Exception:
        pass


def upsert_agent(conn: sqlite3.Connection, data: HeartbeatIn, status: str, health: str) -> None:
    tags_json = json.dumps(data.tags) if data.tags is not None else None
    conn.execute(
        """
        INSERT INTO agents (agent_id, name, os, tags, last_seen, status, last_health, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(agent_id) DO UPDATE SET
            name = COALESCE(excluded.name, agents.name),
            os = COALESCE(excluded.os, agents.os),
            tags = COALESCE(excluded.tags, agents.tags),
            last_seen = excluded.last_seen,
            status = excluded.status,
            last_health = excluded.last_health,
            created_at = COALESCE(agents.created_at, excluded.created_at)
        """,
        (data.agent_id, data.name, data.os, tags_json, utc_now(), status, health, utc_now()),
    )
    conn.commit()


def update_agent_status(conn: sqlite3.Connection, agent_id: str, status: str) -> None:
    conn.execute(
        "UPDATE agents SET status = ?, last_seen = ? WHERE agent_id = ?",
        (status, utc_now(), agent_id),
    )
    conn.commit()


def store_metrics(conn: sqlite3.Connection, metrics: MetricsIn) -> None:
    conn.execute(
        """
        INSERT INTO metrics (
            agent_id, ts, cpu, ram, gpu, temp, disk, net_rx, net_tx, services, extra
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metrics.agent_id,
            parse_ts(metrics.ts).isoformat(),
            metrics.cpu,
            metrics.ram,
            metrics.gpu,
            metrics.temp,
            metrics.disk,
            metrics.net_rx,
            metrics.net_tx,
            json.dumps(metrics.services or {}),
            json.dumps(metrics.extra or {}),
        ),
    )
    conn.commit()


def latest_metrics_for_agent(conn: sqlite3.Connection, agent_id: str) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        """
        SELECT * FROM metrics
        WHERE agent_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (agent_id,),
    ).fetchone()
    if not row:
        return None
    data = dict(row)
    data["services"] = json.loads(data.get("services") or "{}")
    data["extra"] = json.loads(data.get("extra") or "{}")
    return data


@app.on_event("startup")
async def on_startup() -> None:
    init_db()
    asyncio.create_task(offline_watcher())


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("app/static/index.html")


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/api/config")
async def get_config(user: Dict[str, Any] = Depends(require_user)) -> Dict[str, Any]:
    return {
        "heartbeat_timeout": HEARTBEAT_TIMEOUT,
        "thresholds": {
            "cpu": {"warn": CPU_WARN, "crit": CPU_CRIT},
            "ram": {"warn": RAM_WARN, "crit": RAM_CRIT},
            "gpu": {"warn": GPU_WARN, "crit": GPU_CRIT},
            "temp": {"warn": TEMP_WARN, "crit": TEMP_CRIT},
            "disk": {"warn": DISK_WARN, "crit": DISK_CRIT},
        },
    }


@app.post("/api/auth/login")
async def login(payload: LoginIn) -> Dict[str, Any]:
    conn = get_conn()
    row = conn.execute(
        "SELECT username, password_hash, role FROM users WHERE username = ?",
        (payload.username,),
    ).fetchone()
    conn.close()
    if not row or not verify_password(payload.password, row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(row["username"], row["role"])
    return {"access_token": token, "token_type": "bearer"}


@app.get("/api/auth/me")
async def auth_me(user: Dict[str, Any] = Depends(require_user)) -> Dict[str, Any]:
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


@app.get("/api/admin/users")
async def list_users(admin: Dict[str, Any] = Depends(require_admin)) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY username ASC",
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


@app.post("/api/admin/users")
async def create_user(payload: UserCreate, admin: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    role = payload.role.lower()
    if role not in {"admin", "user"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid role")
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (payload.username, hash_password(payload.password), role, utc_now()),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        conn.close()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Username already exists") from exc
    conn.close()
    return {"username": payload.username, "role": role}


@app.delete("/api/admin/users/{user_id}")
async def delete_user(user_id: int, admin: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    if admin["id"] == user_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete yourself")
    conn = get_conn()
    conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.get("/api/admin/agents")
async def admin_agents(admin: Dict[str, Any] = Depends(require_admin)) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT agent_id, name, os, tags, last_seen, status, last_health, created_at
        FROM agents
        ORDER BY name, agent_id
        """
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        agent = dict(row)
        agent["tags"] = json.loads(agent.get("tags") or "[]")
        result.append(agent)
    return result


@app.post("/api/admin/agents")
async def admin_create_agent(
    payload: AgentCreate, admin: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    conn = get_conn()
    exists = conn.execute(
        "SELECT agent_id FROM agents WHERE agent_id = ?",
        (payload.agent_id,),
    ).fetchone()
    if exists:
        conn.close()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Agent already exists")
    token = create_agent_record(conn, payload.agent_id, payload.name, payload.tags)
    conn.close()
    return {"agent_id": payload.agent_id, "token": token}


@app.post("/api/admin/agents/{agent_id}/rotate")
async def admin_rotate_agent(
    agent_id: str, admin: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    conn = get_conn()
    exists = conn.execute(
        "SELECT agent_id FROM agents WHERE agent_id = ?",
        (agent_id,),
    ).fetchone()
    if not exists:
        conn.close()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    token = rotate_agent_token(conn, agent_id)
    conn.close()
    return {"agent_id": agent_id, "token": token}


@app.delete("/api/admin/agents/{agent_id}")
async def admin_delete_agent(
    agent_id: str, admin: Dict[str, Any] = Depends(require_admin)
) -> Dict[str, Any]:
    conn = get_conn()
    conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.post("/api/heartbeat")
async def heartbeat(
    payload: HeartbeatIn,
    background_tasks: BackgroundTasks,
    agent_token: Optional[str] = Header(None, alias="X-Agent-Token"),
) -> Dict[str, Any]:
    conn = get_conn()
    require_agent_auth(conn, payload.agent_id, agent_token)
    current = conn.execute(
        "SELECT status, last_health FROM agents WHERE agent_id = ?",
        (payload.agent_id,),
    ).fetchone()
    status = "online"
    health = current["last_health"] if current else "ok"

    upsert_agent(conn, payload, status, health)

    if payload.services:
        metrics = MetricsIn(agent_id=payload.agent_id, services=payload.services)
        store_metrics(conn, metrics)
        svc_state = services_status(payload.services)
        if svc_state == "down":
            alert = create_alert(
                conn,
                payload.agent_id,
                "crit",
                "service",
                "One or more services are down",
            )
            background_tasks.add_task(send_webhook, alert)

    if current and current["status"] == "offline":
        alert = create_alert(conn, payload.agent_id, "ok", "status", "Agent back online")
        background_tasks.add_task(send_webhook, alert)

    conn.close()
    return {"status": "ok"}


@app.post("/api/metrics")
async def metrics(
    payload: MetricsIn,
    background_tasks: BackgroundTasks,
    agent_token: Optional[str] = Header(None, alias="X-Agent-Token"),
) -> Dict[str, Any]:
    conn = get_conn()
    require_agent_auth(conn, payload.agent_id, agent_token)
    current = conn.execute(
        "SELECT status, last_health FROM agents WHERE agent_id = ?",
        (payload.agent_id,),
    ).fetchone()

    if not current:
        if REQUIRE_AGENT_TOKEN:
            conn.close()
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Agent not registered")
        hb = HeartbeatIn(agent_id=payload.agent_id)
        upsert_agent(conn, hb, "online", "ok")
        current = {"status": "online", "last_health": "ok"}

    store_metrics(conn, payload)
    health = compute_health(payload)
    update_agent_status(conn, payload.agent_id, "online")

    if health != current["last_health"]:
        level = "ok" if health == "ok" else "warn" if health == "warn" else "crit"
        alert = create_alert(
            conn,
            payload.agent_id,
            level,
            "health",
            f"Health changed to {health}",
        )
        background_tasks.add_task(send_webhook, alert)
        conn.execute(
            "UPDATE agents SET last_health = ? WHERE agent_id = ?",
            (health, payload.agent_id),
        )
        conn.commit()

    svc_state = services_status(payload.services)
    if svc_state == "down":
        alert = create_alert(
            conn,
            payload.agent_id,
            "crit",
            "service",
            "One or more services are down",
        )
        background_tasks.add_task(send_webhook, alert)

    conn.close()
    return {"status": "ok", "health": health}


@app.get("/api/agents")
async def list_agents(user: Dict[str, Any] = Depends(require_user)) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT a.*, (
            SELECT id FROM metrics WHERE agent_id = a.agent_id ORDER BY id DESC LIMIT 1
        ) AS metric_id
        FROM agents a
        ORDER BY a.name, a.agent_id
        """
    ).fetchall()
    result: List[Dict[str, Any]] = []
    for row in rows:
        agent = dict(row)
        agent.pop("agent_token", None)
        agent["tags"] = json.loads(agent.get("tags") or "[]")
        latest = None
        if agent.get("metric_id"):
            latest = conn.execute(
                "SELECT * FROM metrics WHERE id = ?",
                (agent["metric_id"],),
            ).fetchone()
            if latest:
                latest = dict(latest)
                latest["services"] = json.loads(latest.get("services") or "{}")
                latest["extra"] = json.loads(latest.get("extra") or "{}")
        agent["latest_metrics"] = latest
        result.append(agent)
    conn.close()
    return result


@app.get("/api/metrics")
async def list_metrics(
    agent_id: str = Query(...),
    since_minutes: int = Query(360, ge=1, le=10080),
    user: Dict[str, Any] = Depends(require_user),
) -> List[Dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    conn = get_conn()
    rows = conn.execute(
        """
        SELECT * FROM metrics
        WHERE agent_id = ? AND ts >= ?
        ORDER BY ts ASC
        """,
        (agent_id, since.isoformat()),
    ).fetchall()
    conn.close()
    result = []
    for row in rows:
        item = dict(row)
        item["services"] = json.loads(item.get("services") or "{}")
        item["extra"] = json.loads(item.get("extra") or "{}")
        result.append(item)
    return result


@app.get("/api/alerts")
async def list_alerts(
    limit: int = Query(25, ge=1, le=200),
    user: Dict[str, Any] = Depends(require_user),
) -> List[Dict[str, Any]]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


async def offline_watcher() -> None:
    while True:
        await asyncio.sleep(OFFLINE_CHECK_INTERVAL)
        conn = get_conn()
        rows = conn.execute("SELECT agent_id, last_seen, status FROM agents").fetchall()
        now = datetime.now(timezone.utc)
        for row in rows:
            last_seen = parse_ts(row["last_seen"])
            if (now - last_seen).total_seconds() > HEARTBEAT_TIMEOUT:
                if row["status"] != "offline":
                    conn.execute(
                        "UPDATE agents SET status = ? WHERE agent_id = ?",
                        ("offline", row["agent_id"]),
                    )
                    alert = create_alert(
                        conn,
                        row["agent_id"],
                        "crit",
                        "status",
                        "Agent went offline",
                    )
                    send_webhook(alert)
        conn.close()
