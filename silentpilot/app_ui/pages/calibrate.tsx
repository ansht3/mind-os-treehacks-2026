/**
 * Calibration screen: record EMG samples for each command label.
 */

import { useState, useCallback } from "react";
import Link from "next/link";
import { useEMGWebSocket } from "../lib/ws";
import { emgApi } from "../lib/api";
import SignalPlot from "../components/SignalPlot";
import PTTButton from "../components/PTTButton";

const COMMANDS = [
  "OPEN", "SEARCH", "CLICK", "SCROLL",
  "TYPE", "ENTER", "CONFIRM", "CANCEL",
];

const USER_ID = "demo1";
const TARGET_SAMPLES = 20;

export default function Calibrate() {
  const { connected, lastRaw, calibProgress, send } = useEMGWebSocket();
  const [selectedLabel, setSelectedLabel] = useState(COMMANDS[0]);
  const [recording, setRecording] = useState(false);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  // Update counts from WS progress messages
  if (
    calibProgress &&
    calibProgress.label &&
    calibProgress.count !== counts[calibProgress.label]
  ) {
    setCounts((prev) => ({
      ...prev,
      [calibProgress.label]: calibProgress.count,
    }));
  }

  const handlePTTStart = useCallback(() => {
    send({ type: "ptt_start", data: { label: selectedLabel } });
  }, [send, selectedLabel]);

  const handlePTTStop = useCallback(() => {
    send({ type: "ptt_stop", data: {} });
  }, [send]);

  const startCalibration = async () => {
    try {
      await emgApi.calibStart(selectedLabel);
      setRecording(true);
      setMessage(`Recording "${selectedLabel}" -- hold PTT and subvocalize`);
    } catch (e) {
      setMessage(`Error: ${e}`);
    }
  };

  const stopCalibration = async () => {
    try {
      await emgApi.calibStop();
      setRecording(false);
      setMessage("Recording stopped");
    } catch (e) {
      setMessage(`Error: ${e}`);
    }
  };

  const saveData = async () => {
    setSaving(true);
    try {
      const result = await emgApi.calibSave(USER_ID);
      setMessage(
        `Saved ${result.new_segments} segments (total: ${result.total_segments})`
      );
    } catch (e) {
      setMessage(`Error: ${e}`);
    }
    setSaving(false);
  };

  const totalSamples = Object.values(counts).reduce((a, b) => a + b, 0);

  return (
    <div className="min-h-screen p-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <Link href="/" className="text-sp-accent text-sm hover:underline">
            &larr; Back
          </Link>
          <h1 className="text-2xl font-bold mt-1">Calibration</h1>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-sp-green" : "bg-sp-red"
            }`}
          />
          <span className="text-sm text-gray-500">
            {connected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>

      {/* Signal Plot */}
      <SignalPlot lastRaw={lastRaw} />

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
        {/* Left: Label selection + PTT */}
        <div className="bg-sp-card border border-sp-border rounded-lg p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">
            Command Label
          </h3>
          <div className="grid grid-cols-4 gap-2 mb-6">
            {COMMANDS.map((cmd) => (
              <button
                key={cmd}
                onClick={() => setSelectedLabel(cmd)}
                className={`px-3 py-2 rounded text-xs font-mono transition-colors ${
                  selectedLabel === cmd
                    ? "bg-sp-accent text-white"
                    : "bg-gray-800 text-gray-400 hover:bg-gray-700"
                }`}
              >
                {cmd}
                {counts[cmd] ? (
                  <span className="ml-1 text-gray-500">({counts[cmd]})</span>
                ) : null}
              </button>
            ))}
          </div>

          <div className="flex flex-col items-center gap-4">
            <PTTButton
              label={selectedLabel}
              onStart={handlePTTStart}
              onStop={handlePTTStop}
              disabled={!connected || !recording}
            />

            <div className="flex gap-3">
              {!recording ? (
                <button
                  onClick={startCalibration}
                  disabled={!connected}
                  className="px-4 py-2 bg-sp-accent text-white rounded hover:bg-sp-accent/80 disabled:opacity-50 text-sm"
                >
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={stopCalibration}
                  className="px-4 py-2 bg-sp-red text-white rounded hover:bg-sp-red/80 text-sm"
                >
                  Stop Recording
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Right: Progress + Save */}
        <div className="bg-sp-card border border-sp-border rounded-lg p-6">
          <h3 className="text-sm font-medium text-gray-400 mb-3">Progress</h3>

          <div className="space-y-2 mb-6">
            {COMMANDS.map((cmd) => {
              const count = counts[cmd] || 0;
              const pct = Math.min(100, (count / TARGET_SAMPLES) * 100);
              return (
                <div key={cmd} className="flex items-center gap-3">
                  <span className="text-xs font-mono w-16 text-gray-400">
                    {cmd}
                  </span>
                  <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-sp-accent rounded-full transition-all"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono text-gray-500 w-12">
                    {count}/{TARGET_SAMPLES}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="text-center">
            <p className="text-sm text-gray-400 mb-3">
              Total: {totalSamples} samples
            </p>
            <button
              onClick={saveData}
              disabled={totalSamples === 0 || saving}
              className="px-6 py-2 bg-sp-green text-white rounded hover:bg-sp-green/80 disabled:opacity-50 text-sm"
            >
              {saving ? "Saving..." : "Save Calibration Data"}
            </button>
          </div>

          {message && (
            <p className="mt-4 text-xs text-gray-400 text-center">{message}</p>
          )}
        </div>
      </div>
    </div>
  );
}
