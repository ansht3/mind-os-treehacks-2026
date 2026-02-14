/**
 * WebSocket hook for connecting to EMG Core.
 */

import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = "ws://localhost:8000/ws/live";

export interface WSMessage {
  type: string;
  data: Record<string, unknown>;
}

export function useEMGWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastRaw, setLastRaw] = useState<{ t: number; ch: number[] } | null>(null);
  const [lastPrediction, setLastPrediction] = useState<{
    cmd: string;
    p: number;
    t: number;
  } | null>(null);
  const [calibProgress, setCalibProgress] = useState<{
    label: string;
    count: number;
  } | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      console.log("[WS] Connected to EMG Core");
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);

        switch (msg.type) {
          case "raw":
            setLastRaw(msg.data as { t: number; ch: number[] });
            break;
          case "prediction":
            setLastPrediction(
              msg.data as { cmd: string; p: number; t: number }
            );
            break;
          case "calib_progress":
            setCalibProgress(
              msg.data as { label: string; count: number }
            );
            break;
        }
      } catch {}
    };

    ws.onclose = () => {
      setConnected(false);
      console.log("[WS] Disconnected. Reconnecting in 2s...");
      setTimeout(connect, 2000);
    };

    ws.onerror = () => {
      ws.close();
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  const send = useCallback((msg: WSMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return {
    connected,
    lastRaw,
    lastPrediction,
    calibProgress,
    send,
  };
}
