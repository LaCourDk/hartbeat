# hartbeat

Hartbeat is a lightweight heartbeat + metrics server with a simple web UI for live status, charts, and alerts. It stores metrics in SQLite and exposes a small JSON API.

## Run (Docker Desktop)

```bash
docker compose up --build
```

Open `http://localhost:9090`.

Default login (change in `docker-compose.yml`):
- User: `admin`
- Password: `admin123`

## Admin + users

After login, the Admin panel lets you:
- Create sub-users (user/admin)
- Create agents with a name + token
- Rotate tokens

## Agent install

Create an agent token in the Admin panel and use it on your machines.

Linux/Mac (bash):
```bash
cd agent
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
SERVER_URL=http://<server-ip>:9090 AGENT_ID=my-host AGENT_NAME=my-host AGENT_TOKEN=<token> python agent.py
```

Windows (PowerShell):
```powershell
cd agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:SERVER_URL="http://<server-ip>:9090"; $env:AGENT_ID="my-host"; $env:AGENT_NAME="my-host"; $env:AGENT_TOKEN="<token>"; python agent.py
```

Notes:
- `SERVICES` checks for running process names and sends `ok` or `down`.
- `INTERVAL` controls send frequency in seconds (default 10).
- `HEARTBEAT_EVERY` controls heartbeat interval (default 60).

## API quick test

Create an agent in the Admin panel to get a token, then:

```bash
curl -X POST http://localhost:9090/api/heartbeat \
  -H 'Content-Type: application/json' \
  -H 'X-Agent-Token: <token>' \
  -d '{"agent_id":"demo","name":"Demo Host","tags":["lab"],"services":{"nginx":"ok"}}'

curl -X POST http://localhost:9090/api/metrics \
  -H 'Content-Type: application/json' \
  -H 'X-Agent-Token: <token>' \
  -d '{"agent_id":"demo","cpu":23.4,"ram":54.2,"gpu":10.1,"temp":49.5,"disk":72.2}'
```

## Configuration

You can adjust alert thresholds and timeouts via environment variables in `docker-compose.yml`.

- `ADMIN_USER`, `ADMIN_PASSWORD`
- `JWT_SECRET` (set for stable logins)
- `REQUIRE_AGENT_TOKEN` (default `1`)
- `HEARTBEAT_TIMEOUT`
- `CPU_WARN`, `CPU_CRIT`
- `RAM_WARN`, `RAM_CRIT`
- `GPU_WARN`, `GPU_CRIT`
- `TEMP_WARN`, `TEMP_CRIT`
- `DISK_WARN`, `DISK_CRIT`
- `WEBHOOK_URL` for alert webhooks
