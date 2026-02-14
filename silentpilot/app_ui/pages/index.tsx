/**
 * Home / navigation page for SilentPilot UI.
 */

import Link from "next/link";
import { useEMGWebSocket } from "../lib/ws";

export default function Home() {
  const { connected } = useEMGWebSocket();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-8">
      <div className="text-center mb-12">
        <h1 className="text-5xl font-bold mb-3">
          Silent<span className="text-sp-accent">Pilot</span>
        </h1>
        <p className="text-gray-400 text-lg">
          Control your computer with silent speech
        </p>
        <div className="mt-4 flex items-center justify-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              connected ? "bg-sp-green" : "bg-sp-red"
            }`}
          />
          <span className="text-sm text-gray-500">
            EMG Core: {connected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-3xl w-full">
        <Link
          href="/calibrate"
          className="bg-sp-card border border-sp-border rounded-xl p-6 hover:border-sp-accent/50 transition-colors group"
        >
          <h2 className="text-xl font-semibold mb-2 group-hover:text-sp-accent transition-colors">
            1. Calibrate
          </h2>
          <p className="text-sm text-gray-400">
            Record command samples and build your personal EMG profile.
          </p>
        </Link>

        <Link
          href="/train"
          className="bg-sp-card border border-sp-border rounded-xl p-6 hover:border-sp-accent/50 transition-colors group"
        >
          <h2 className="text-xl font-semibold mb-2 group-hover:text-sp-accent transition-colors">
            2. Train
          </h2>
          <p className="text-sm text-gray-400">
            Train your personal classifier and review accuracy metrics.
          </p>
        </Link>

        <Link
          href="/demo"
          className="bg-sp-card border border-sp-border rounded-xl p-6 hover:border-sp-accent/50 transition-colors group"
        >
          <h2 className="text-xl font-semibold mb-2 group-hover:text-sp-accent transition-colors">
            3. Demo
          </h2>
          <p className="text-sm text-gray-400">
            Live silent speech to computer control with AI agent.
          </p>
        </Link>
      </div>
    </div>
  );
}
