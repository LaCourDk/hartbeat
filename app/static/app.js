const state = {
  agents: [],
  selectedId: null,
  config: null,
  charts: {},
  user: null,
  timersStarted: false,
  adminBound: false,
  chartsInitialized: false,
};

const tokenKey = "hb_token";

const colors = {
  cpu: "#29d7a1",
  ram: "#f0b429",
  temp: "#ff5c5c",
  gpu: "#7fd8ff",
  disk: "#f08d29",
  netRx: "#29d7a1",
  netTx: "#f0b429",
};

function getToken() {
  return localStorage.getItem(tokenKey);
}

function setToken(token) {
  if (token) {
    localStorage.setItem(tokenKey, token);
  } else {
    localStorage.removeItem(tokenKey);
  }
}

function showLogin(show) {
  const login = document.getElementById("loginScreen");
  const app = document.getElementById("appRoot");
  login.style.display = show ? "flex" : "none";
  app.style.display = show ? "none" : "flex";
}

async function apiFetch(url, options = {}) {
  const token = getToken();
  const headers = Object.assign({}, options.headers || {});
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  const res = await fetch(url, { ...options, headers });
  if (res.status === 401) {
    showLogin(true);
  }
  return res;
}

async function fetchJSON(url, options) {
  const res = await apiFetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed: ${res.status}`);
  }
  return res.json();
}

function levelFor(value, warn, crit) {
  if (value === null || value === undefined) return "offline";
  if (value >= crit) return "crit";
  if (value >= warn) return "warn";
  return "ok";
}

function formatValue(value, unit = "%") {
  if (value === null || value === undefined) return "--";
  return `${value.toFixed(1)}${unit}`;
}

function statusPill(status) {
  const cls = status === "offline" ? "status-offline" : `status-${status}`;
  return `<span class="status-pill ${cls}">${status}</span>`;
}

function renderAgents() {
  const list = document.getElementById("agentList");
  list.innerHTML = "";
  if (!state.agents.length) {
    list.innerHTML = "<div class=\"footer\">No agents yet. Create one in Admin.</div>";
    return;
  }

  state.agents.forEach((agent) => {
    const div = document.createElement("div");
    div.className = `agent-item ${agent.agent_id === state.selectedId ? "active" : ""}`;
    div.onclick = () => {
      state.selectedId = agent.agent_id;
      renderAgents();
      renderSelected(agent);
      loadMetrics(agent.agent_id);
    };

    const status = agent.status === "offline" ? "offline" : agent.last_health || "ok";
    const tags = (agent.tags || []).join(", ") || "no tags";

    div.innerHTML = `
      <div class="agent-meta">
        <div class="agent-name">${agent.name || agent.agent_id}</div>
        <div class="agent-tags">${tags}</div>
      </div>
      ${statusPill(status)}
    `;

    list.appendChild(div);
  });
}

function summaryCard(title, value, status, barValue) {
  const bar = Number.isFinite(barValue)
    ? `<div class="bar"><span style="width: ${Math.min(barValue, 100)}%"></span></div>`
    : "";
  return `
    <div class="summary-card">
      <h3>${title}</h3>
      <div class="summary-value">${value}</div>
      <div class="summary-sub">${status}</div>
      ${bar}
    </div>
  `;
}

function renderSelected(agent) {
  const title = document.getElementById("agentTitle");
  const summary = document.getElementById("summaryCards");

  if (!agent) {
    title.textContent = "Select an agent";
    summary.innerHTML = "";
    return;
  }

  const latest = agent.latest_metrics || {};
  const thresholds = state.config.thresholds;

  const cpuStatus = levelFor(latest.cpu, thresholds.cpu.warn, thresholds.cpu.crit);
  const ramStatus = levelFor(latest.ram, thresholds.ram.warn, thresholds.ram.crit);
  const gpuStatus = levelFor(latest.gpu, thresholds.gpu.warn, thresholds.gpu.crit);
  const tempStatus = levelFor(latest.temp, thresholds.temp.warn, thresholds.temp.crit);
  const diskStatus = levelFor(latest.disk, thresholds.disk.warn, thresholds.disk.crit);

  title.textContent = `${agent.name || agent.agent_id} (${agent.status})`;

  summary.innerHTML = [
    summaryCard("CPU", formatValue(latest.cpu), cpuStatus, latest.cpu),
    summaryCard("RAM", formatValue(latest.ram), ramStatus, latest.ram),
    summaryCard("GPU", formatValue(latest.gpu), gpuStatus, latest.gpu),
    summaryCard("Temp", formatValue(latest.temp, "C"), tempStatus, latest.temp),
    summaryCard("Disk", formatValue(latest.disk), diskStatus, latest.disk),
  ].join("");
}

function dataset(label, color) {
  return {
    label,
    data: [],
    borderColor: color,
    backgroundColor: "rgba(255,255,255,0.05)",
    tension: 0.32,
    pointRadius: 0,
    borderWidth: 2,
  };
}

function buildChart(ctx, datasets, showLegend = false) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { color: "#b3c6bf" },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          ticks: { color: "#b3c6bf" },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
      },
      plugins: {
        legend: { display: showLegend },
      },
    },
  });
}

function initCharts() {
  if (state.chartsInitialized) return;
  state.chartsInitialized = true;
  state.charts.cpu = buildChart(document.getElementById("cpuChart"), [dataset("CPU", colors.cpu)]);
  state.charts.ram = buildChart(document.getElementById("ramChart"), [dataset("RAM", colors.ram)]);
  state.charts.temp = buildChart(document.getElementById("tempChart"), [dataset("Temp", colors.temp)]);
  state.charts.gpu = buildChart(document.getElementById("gpuChart"), [dataset("GPU", colors.gpu)]);
  state.charts.disk = buildChart(document.getElementById("diskChart"), [dataset("Disk", colors.disk)]);
  state.charts.net = buildChart(
    document.getElementById("netChart"),
    [dataset("RX", colors.netRx), dataset("TX", colors.netTx)],
    true
  );
}

function updateChart(chart, labels, series) {
  chart.data.labels = labels;
  series.forEach((data, idx) => {
    chart.data.datasets[idx].data = data;
  });
  chart.update();
}

function computeRate(rows, key) {
  if (!rows.length) return [];
  const rates = [0];
  for (let i = 1; i < rows.length; i += 1) {
    const prev = rows[i - 1];
    const curr = rows[i];
    const delta = (curr[key] ?? 0) - (prev[key] ?? 0);
    const dt = (new Date(curr.ts) - new Date(prev.ts)) / 1000;
    const rate = dt > 0 ? Math.max(delta / 1024 / dt, 0) : 0;
    rates.push(rate);
  }
  return rates;
}

async function loadMetrics(agentId) {
  if (!agentId) return;
  const rows = await fetchJSON(`/api/metrics?agent_id=${encodeURIComponent(agentId)}&since_minutes=360`);
  const labels = rows.map((row) => new Date(row.ts).toLocaleTimeString());

  updateChart(state.charts.cpu, labels, [rows.map((row) => row.cpu)]);
  updateChart(state.charts.ram, labels, [rows.map((row) => row.ram)]);
  updateChart(state.charts.temp, labels, [rows.map((row) => row.temp)]);
  updateChart(state.charts.gpu, labels, [rows.map((row) => row.gpu)]);
  updateChart(state.charts.disk, labels, [rows.map((row) => row.disk)]);
  updateChart(state.charts.net, labels, [computeRate(rows, "net_rx"), computeRate(rows, "net_tx")]);
}

async function loadAlerts() {
  const alerts = await fetchJSON("/api/alerts?limit=8");
  const list = document.getElementById("alertList");
  if (!alerts.length) {
    list.innerHTML = "<div class=\"footer\">No alerts yet.</div>";
    return;
  }
  list.innerHTML = alerts
    .map((alert) => {
      return `
        <div class="alert-item">
          <div>${alert.message}</div>
          <span>${new Date(alert.ts).toLocaleString()}</span>
        </div>
      `;
    })
    .join("");
}

function updateUserUI() {
  const el = document.getElementById("currentUser");
  if (!state.user) {
    el.textContent = "-";
    return;
  }
  el.textContent = `${state.user.username} (${state.user.role})`;
  const adminPanel = document.getElementById("adminPanel");
  adminPanel.style.display = state.user.role === "admin" ? "block" : "none";
}

async function loadUser() {
  state.user = await fetchJSON("/api/auth/me");
  updateUserUI();
}

async function refresh() {
  state.agents = await fetchJSON("/api/agents");
  if (!state.selectedId && state.agents.length) {
    state.selectedId = state.agents[0].agent_id;
  }
  renderAgents();
  const selected = state.agents.find((agent) => agent.agent_id === state.selectedId);
  if (selected) {
    renderSelected(selected);
    await loadMetrics(selected.agent_id);
  }
  await loadAlerts();
}

async function loadAdminData() {
  if (!state.user || state.user.role !== "admin") return;
  const users = await fetchJSON("/api/admin/users");
  const agents = await fetchJSON("/api/admin/agents");

  const userList = document.getElementById("userList");
  userList.innerHTML = users
    .map((user) => {
      const disable = user.id === state.user.id ? "disabled" : "";
      return `
        <div class="list-item">
          <div>${user.username} (${user.role})</div>
          <button class="ghost" data-user="${user.id}" ${disable}>Delete</button>
        </div>
      `;
    })
    .join("");

  userList.querySelectorAll("button[data-user]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-user");
      await fetchJSON(`/api/admin/users/${id}`, { method: "DELETE" });
      await loadAdminData();
    });
  });

  const agentList = document.getElementById("adminAgentList");
  agentList.innerHTML = agents
    .map((agent) => {
      return `
        <div class="list-item">
          <div>${agent.name || agent.agent_id}</div>
          <button class="ghost" data-agent="${agent.agent_id}">Rotate token</button>
        </div>
      `;
    })
    .join("");

  agentList.querySelectorAll("button[data-agent]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = btn.getAttribute("data-agent");
      const result = await fetchJSON(`/api/admin/agents/${id}/rotate`, { method: "POST" });
      renderAgentToken(result.agent_id, result.token);
    });
  });
}

function renderAgentToken(agentId, token) {
  const box = document.getElementById("agentTokenBox");
  const serverUrl = window.location.origin;
  box.textContent = `Agent: ${agentId}\nToken: ${token}\n\nLinux/Mac:\nSERVER_URL=${serverUrl} AGENT_ID=${agentId} AGENT_TOKEN=${token} python agent.py\n\nWindows (PowerShell):\n$env:SERVER_URL=\"${serverUrl}\"; $env:AGENT_ID=\"${agentId}\"; $env:AGENT_TOKEN=\"${token}\"; python agent.py`;
}

function bindAdminForms() {
  if (state.adminBound) return;
  state.adminBound = true;
  const userForm = document.getElementById("createUserForm");
  userForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const username = document.getElementById("newUserName").value.trim();
    const password = document.getElementById("newUserPass").value.trim();
    const role = document.getElementById("newUserRole").value;
    const msg = document.getElementById("userCreateMsg");
    msg.textContent = "";
    if (!username || !password) return;
    try {
      await fetchJSON("/api/admin/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password, role }),
      });
      msg.textContent = "User created";
      userForm.reset();
      await loadAdminData();
    } catch (err) {
      msg.textContent = err.message;
    }
  });

  const agentForm = document.getElementById("createAgentForm");
  agentForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const agentId = document.getElementById("newAgentId").value.trim();
    const name = document.getElementById("newAgentName").value.trim();
    const tagsRaw = document.getElementById("newAgentTags").value.trim();
    const tags = tagsRaw ? tagsRaw.split(",").map((t) => t.trim()).filter(Boolean) : [];
    if (!agentId) return;
    const result = await fetchJSON("/api/admin/agents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agent_id: agentId, name: name || null, tags }),
    });
    renderAgentToken(result.agent_id, result.token);
    agentForm.reset();
    await loadAdminData();
  });
}

function bindLogin() {
  const form = document.getElementById("loginForm");
  const error = document.getElementById("loginError");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    error.textContent = "";
    const username = document.getElementById("loginUser").value.trim();
    const password = document.getElementById("loginPass").value.trim();
    try {
      const result = await fetchJSON("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      setToken(result.access_token);
      showLogin(false);
      await bootstrap();
      startIntervals();
    } catch (err) {
      error.textContent = "Login failed";
    }
  });

  const logout = document.getElementById("logoutBtn");
  logout.addEventListener("click", () => {
    setToken(null);
    showLogin(true);
  });
}

function startIntervals() {
  if (state.timersStarted) return;
  state.timersStarted = true;
  setInterval(refresh, 10000);
  setInterval(loadAdminData, 20000);
}

async function bootstrap() {
  await loadUser();
  state.config = await fetchJSON("/api/config");
  initCharts();
  bindAdminForms();
  await refresh();
  await loadAdminData();
}

async function init() {
  bindLogin();
  const token = getToken();
  if (!token) {
    showLogin(true);
    return;
  }
  showLogin(false);
  await bootstrap();
  startIntervals();
}

init().catch((err) => {
  console.error(err);
});
