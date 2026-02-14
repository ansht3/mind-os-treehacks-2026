/**
 * Push-to-Talk button component.
 *
 * Supports both mouse hold and spacebar hold.
 * Sends ptt_start / ptt_stop WebSocket messages.
 */

import { useCallback, useEffect, useState } from "react";

interface PTTButtonProps {
  label: string;
  onStart: () => void;
  onStop: () => void;
  disabled?: boolean;
}

export default function PTTButton({
  label,
  onStart,
  onStop,
  disabled = false,
}: PTTButtonProps) {
  const [active, setActive] = useState(false);

  const handleStart = useCallback(() => {
    if (disabled) return;
    setActive(true);
    onStart();
  }, [disabled, onStart]);

  const handleStop = useCallback(() => {
    if (!active) return;
    setActive(false);
    onStop();
  }, [active, onStop]);

  // Spacebar PTT
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && !e.repeat && !disabled) {
        e.preventDefault();
        handleStart();
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        e.preventDefault();
        handleStop();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [disabled, handleStart, handleStop]);

  return (
    <button
      onMouseDown={handleStart}
      onMouseUp={handleStop}
      onMouseLeave={handleStop}
      disabled={disabled}
      className={`
        w-40 h-40 rounded-full border-4 transition-all duration-150
        flex flex-col items-center justify-center select-none
        ${
          active
            ? "bg-sp-accent/30 border-sp-accent scale-95 shadow-lg shadow-sp-accent/20"
            : disabled
            ? "bg-gray-800 border-gray-700 text-gray-600 cursor-not-allowed"
            : "bg-sp-card border-sp-border hover:border-sp-accent/50 cursor-pointer"
        }
      `}
    >
      <span className="text-2xl font-bold">{active ? "REC" : "PTT"}</span>
      <span className="text-xs text-gray-400 mt-1">
        {active ? "Release to stop" : "Hold or Space"}
      </span>
      <span className="text-xs text-sp-accent mt-1 font-medium">{label}</span>
    </button>
  );
}
