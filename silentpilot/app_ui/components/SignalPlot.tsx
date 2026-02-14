/**
 * Real-time 4-channel sEMG signal plot.
 */

import { useEffect, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
} from "recharts";

const CHANNEL_COLORS = ["#6366f1", "#22c55e", "#eab308", "#ef4444"];
const MAX_POINTS = 150;

interface SignalPlotProps {
  lastRaw: { t: number; ch: number[] } | null;
}

export default function SignalPlot({ lastRaw }: SignalPlotProps) {
  const [data, setData] = useState<Record<string, number>[]>([]);

  useEffect(() => {
    if (!lastRaw) return;

    setData((prev) => {
      const next = [
        ...prev,
        {
          t: prev.length,
          ch0: lastRaw.ch[0] || 0,
          ch1: lastRaw.ch[1] || 0,
          ch2: lastRaw.ch[2] || 0,
          ch3: lastRaw.ch[3] || 0,
        },
      ];
      return next.slice(-MAX_POINTS);
    });
  }, [lastRaw]);

  return (
    <div className="bg-sp-card border border-sp-border rounded-lg p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-2">
        Live EMG Signal (4 channels)
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data}>
          <XAxis dataKey="t" hide />
          <YAxis domain={[0, 4095]} hide />
          {["ch0", "ch1", "ch2", "ch3"].map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={CHANNEL_COLORS[i]}
              dot={false}
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <div className="flex gap-4 mt-2 text-xs text-gray-500">
        {["Ch 0", "Ch 1", "Ch 2", "Ch 3"].map((label, i) => (
          <span key={label} className="flex items-center gap-1">
            <span
              className="w-2 h-2 rounded-full inline-block"
              style={{ backgroundColor: CHANNEL_COLORS[i] }}
            />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
