/**
 * SilentPilot Agent entry point.
 *
 * Connects to the EMG Core FastAPI WebSocket to receive predictions,
 * routes commands, and drives the orchestrator.
 *
 * Also exposes a simple HTTP API for the frontend to set goals and
 * get agent state.
 */

import "dotenv/config";
import WebSocket from "ws";
import express from "express";
import { createInitialState, AgentState } from "./state.js";
import { routeCommand, CommandEvent } from "./command_router.js";
import { Orchestrator } from "./orchestrator.js";

const EMG_WS_URL = process.env.EMG_WS_URL || "ws://localhost:8000/ws/live";
const AGENT_PORT = parseInt(process.env.AGENT_PORT || "9000", 10);

// --- State ---
let agentState = createInitialState();
let orchestrator = new Orchestrator(agentState);
let ws: WebSocket | null = null;

// --- WebSocket connection to EMG Core ---

function connectToEMG(): void {
  console.log(`[Agent] Connecting to EMG Core at ${EMG_WS_URL}...`);
  ws = new WebSocket(EMG_WS_URL);

  ws.on("open", () => {
    console.log("[Agent] Connected to EMG Core WebSocket");
  });

  ws.on("message", async (data) => {
    try {
      const msg = JSON.parse(data.toString());

      if (msg.type === "prediction" && agentState.active) {
        const prediction = msg.data;
        console.log(
          `[Agent] EMG Command: ${prediction.cmd} (p=${prediction.p.toFixed(2)})`
        );

        const event: CommandEvent = {
          cmd: prediction.cmd,
          confidence: prediction.p,
          mode: "AGENT", // can be overridden by router
        };

        const routed = routeCommand(event);
        console.log(`[Agent] Routed: ${routed.command} -> ${routed.mode}`);

        const result = await orchestrator.execute(routed);
        console.log(`[Agent] Result: ${result.action}`);

        if (result.toolCalls.length > 0) {
          for (const tc of result.toolCalls) {
            console.log(`  Tool: ${tc.tool}(${JSON.stringify(tc.args)})`);
          }
        }
      }
    } catch (e) {
      // Ignore non-prediction messages
    }
  });

  ws.on("close", () => {
    console.log("[Agent] EMG WebSocket disconnected. Reconnecting in 3s...");
    setTimeout(connectToEMG, 3000);
  });

  ws.on("error", (err) => {
    console.error("[Agent] WebSocket error:", err.message);
  });
}

// --- HTTP API for frontend ---

const app = express();
app.use(express.json());

// CORS
app.use((_req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Content-Type");
  res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  next();
});

app.get("/state", (_req, res) => {
  res.json(agentState);
});

app.post("/goal", (req, res) => {
  const { goal } = req.body;
  if (!goal) {
    res.status(400).json({ error: "goal is required" });
    return;
  }
  orchestrator.setGoal(goal);
  console.log(`[Agent] New goal: ${goal}`);
  res.json({ status: "ok", goal });
});

app.post("/command", async (req, res) => {
  /**
   * Manual command injection (for testing without EMG).
   * POST /command { "cmd": "OPEN", "confidence": 1.0 }
   */
  const { cmd, confidence = 1.0 } = req.body;
  if (!cmd) {
    res.status(400).json({ error: "cmd is required" });
    return;
  }

  const event: CommandEvent = { cmd, confidence, mode: "AGENT" };
  const routed = routeCommand(event);
  const result = await orchestrator.execute(routed);

  res.json({
    action: result.action,
    result: result.result,
    toolCalls: result.toolCalls,
    state: agentState,
  });
});

app.post("/reset", (_req, res) => {
  agentState = createInitialState();
  orchestrator = new Orchestrator(agentState);
  res.json({ status: "reset" });
});

// --- Start ---

app.listen(AGENT_PORT, () => {
  console.log(`[Agent] HTTP API on http://localhost:${AGENT_PORT}`);
  console.log(`[Agent] POST /goal to set a task goal`);
  console.log(`[Agent] POST /command to inject a command manually`);
  console.log(`[Agent] GET /state to see agent state`);
});

connectToEMG();
