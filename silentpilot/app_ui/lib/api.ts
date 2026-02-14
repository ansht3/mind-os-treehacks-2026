/**
 * HTTP API client for EMG Core and Agent.
 */

const EMG_BASE = "http://localhost:8000";
const AGENT_BASE = "http://localhost:9000";

async function post(url: string, body?: object) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json();
}

async function get(url: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

// --- EMG Core ---

export const emgApi = {
  calibStart: (label: string) =>
    post(`${EMG_BASE}/calib/start`, { label }),

  calibStop: () =>
    post(`${EMG_BASE}/calib/stop`),

  calibSave: (userId: string) =>
    post(`${EMG_BASE}/calib/save`, { user_id: userId }),

  train: (userId: string) =>
    post(`${EMG_BASE}/train`, { user_id: userId }),

  inferStart: (userId: string) =>
    post(`${EMG_BASE}/infer/start`, { user_id: userId }),

  inferStop: () =>
    post(`${EMG_BASE}/infer/stop`),

  status: () =>
    get(`${EMG_BASE}/status`),
};

// --- Agent ---

export const agentApi = {
  getState: () =>
    get(`${AGENT_BASE}/state`),

  setGoal: (goal: string) =>
    post(`${AGENT_BASE}/goal`, { goal }),

  sendCommand: (cmd: string, confidence = 1.0) =>
    post(`${AGENT_BASE}/command`, { cmd, confidence }),

  reset: () =>
    post(`${AGENT_BASE}/reset`),
};
