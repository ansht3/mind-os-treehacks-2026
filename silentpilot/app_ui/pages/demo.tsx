/**
 * Live Demo screen: silent speech commands + AI agent overlay.
 */

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { useEMGWebSocket } from "../lib/ws";
import { emgApi, agentApi } from "../lib/api";
import SignalPlot from "../components/SignalPlot";
import CommandOverlay from "../components/CommandOverlay";

const COMMANDS = [
  "OPEN", "SEARCH", "CLICK", "SCROLL",
  "TYPE", "ENTER", "CONFIRM", "CANCEL",
];

const USER_ID = "demo1";

export default function Demo() {
  const { connected, lastRaw, lastPrediction, send } = useEMGWebSocket();
  const [inferring, setInferring] = useState(false);
  const [agentState, setAgentState] = useState<any>(null);
  const [goal, setGoal] = useState("");
  const [message, setMessage] = useState("");

  // Poll agent state
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const state = await agentApi.getState();
        setAgentState(state);
      } catch {}
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const startInference = async () => {
    try {
      await emgApi.inferStart(USER_ID);
      setInferring(true);
      setMessage("Inference running -- subvocalize commands!");
    } catch (e) {
      setMessage(`Error: ${e}`);
    }
  };

  const stopInference = async () => {
    try {
      await emgApi.inferStop();
      setInferring(false);
      setMessage("Inference stopped");
    } catch (e) {
      setMessage(`Error: ${e}`);
    }
  };

  const setAgentGoal = async () => {
    if (!goal.trim()) return;
    try {
      await agentApi.setGoal(goal);
      setMessage(`Goal set: ${goal}`);
    } catch (e) {
      setMessage(`Error: ${e}`);
    }
  };

  const handlePTTStart = useCallback(() => {
    send({ type: "ptt_start", data: {} });
  }, [send]);

  const handlePTTStop = useCallback(() => {
    send({ type: "ptt_stop", data: {} });
  }, [send]);

  // Manual command injection (for testing)
  const sendCommand = async (cmd: string) => {
    try {
      const result = await agentApi.sendCommand(cmd);
      setMessage(`${cmd}: ${result.result?.substring(0, 100)}`);
    } catch (e) {
      setMessage(`Error: ${e}`);
    }
  };

  return (
    <div className="min-h-screen p-6 max-w-6xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link href="/" className="text-sp-accent text-sm hover:underline">
            &larr; Back
          </Link>
          <h1 className="text-2xl font-bold mt-1">Live Demo</h1>
        </div>
        <div className="flex items-center gap-4">
          <span
            className={`flex items-center gap-1.5 text-sm ${
              connected ? "text-sp-green" : "text-sp-red"
            }`}
          >
            <span className="w-2 h-2 rounded-full bg-current" />
            EMG {connected ? "OK" : "OFF"}
          </span>
          {inferring ? (
            <button
              onClick={stopInference}
              className="px-4 py-1.5 bg-sp-red text-white rounded text-sm"
            >
              Stop Inference
            </button>
          ) : (
            <button
              onClick={startInference}
              disabled={!connected}
              className="px-4 py-1.5 bg-sp-green text-white rounded text-sm disabled:opacity-50"
            >
              Start Inference
            </button>
          )}
        </div>
      </div>

      {/* Goal Input */}
      <div className="bg-sp-card border border-sp-border rounded-lg p-4 mb-4">
        <label className="text-xs text-gray-500 uppercase tracking-wider block mb-2">
          Task Goal (for the agent)
        </label>
        <div className="flex gap-2">
          <input
            type="text"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="e.g. Find the MIT CSAIL homepage and open the publications page"
            className="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-2 text-sm text-white placeholder-gray-600 focus:border-sp-accent focus:outline-none"
            onKeyDown={(e) => e.key === "Enter" && setAgentGoal()}
          />
          <button
            onClick={setAgentGoal}
            className="px-4 py-2 bg-sp-accent text-white rounded text-sm hover:bg-sp-accent/80"
          >
            Set Goal
          </button>
        </div>
      </div>

      {/* Signal Plot */}
      <SignalPlot lastRaw={lastRaw} />

      {/* Command Overlay */}
      <div className="mt-4">
        <CommandOverlay
          lastPrediction={lastPrediction}
          agentState={agentState}
        />
      </div>

      {/* Manual Command Buttons (for testing) */}
      <div className="bg-sp-card border border-sp-border rounded-lg p-4 mt-4">
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
          Manual Commands (for testing without EMG)
        </p>
        <div className="flex flex-wrap gap-2">
          {COMMANDS.map((cmd) => (
            <button
              key={cmd}
              onClick={() => sendCommand(cmd)}
              className="px-3 py-1.5 bg-gray-800 text-gray-300 rounded text-xs font-mono hover:bg-gray-700 transition-colors"
            >
              {cmd}
            </button>
          ))}
        </div>
      </div>

      {/* Message */}
      {message && (
        <p className="mt-4 text-xs text-gray-400 text-center">{message}</p>
      )}
    </div>
  );
}
