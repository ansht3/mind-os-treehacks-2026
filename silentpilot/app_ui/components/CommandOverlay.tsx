/**
 * Live demo overlay showing recognized commands and agent state.
 */

interface CommandOverlayProps {
  lastPrediction: { cmd: string; p: number; t: number } | null;
  agentState: {
    goal: string;
    plan: string[];
    currentStep: number;
    lastCommand: string | null;
    toolHistory: { tool: string; args: Record<string, unknown>; result: string }[];
    active: boolean;
  } | null;
}

export default function CommandOverlay({
  lastPrediction,
  agentState,
}: CommandOverlayProps) {
  const confidence = lastPrediction?.p ?? 0;
  const confPercent = Math.round(confidence * 100);

  return (
    <div className="space-y-4">
      {/* Recognized Command */}
      <div className="bg-sp-card border border-sp-border rounded-lg p-6 text-center">
        <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
          Recognized Command
        </p>
        <p className="text-5xl font-bold font-mono tracking-tight">
          {lastPrediction?.cmd ?? "---"}
        </p>
        <div className="mt-4 flex items-center justify-center gap-3">
          <span className="text-sm text-gray-400">Confidence:</span>
          <div className="w-48 h-3 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-300"
              style={{
                width: `${confPercent}%`,
                backgroundColor:
                  confPercent >= 85
                    ? "#22c55e"
                    : confPercent >= 75
                    ? "#eab308"
                    : "#ef4444",
              }}
            />
          </div>
          <span className="text-sm font-mono text-gray-300 w-12">
            {confPercent}%
          </span>
        </div>
      </div>

      {/* Agent State */}
      {agentState && (
        <div className="grid grid-cols-2 gap-4">
          {/* Goal & Plan */}
          <div className="bg-sp-card border border-sp-border rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
              Agent Goal
            </p>
            <p className="text-sm font-medium mb-3">
              {agentState.goal || "No goal set"}
            </p>
            {agentState.plan.length > 0 && (
              <div className="space-y-1">
                {agentState.plan.map((step, i) => (
                  <div
                    key={i}
                    className={`text-xs px-2 py-1 rounded ${
                      i === agentState.currentStep
                        ? "bg-sp-accent/20 text-sp-accent"
                        : i < agentState.currentStep
                        ? "text-gray-600 line-through"
                        : "text-gray-400"
                    }`}
                  >
                    {i + 1}. {step}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Last Tool Call */}
          <div className="bg-sp-card border border-sp-border rounded-lg p-4">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
              Last Tool Call
            </p>
            {agentState.toolHistory.length > 0 ? (
              <div className="space-y-2">
                {agentState.toolHistory.slice(-3).map((tc, i) => (
                  <div
                    key={i}
                    className="text-xs bg-gray-900 rounded p-2 font-mono"
                  >
                    <span className="text-sp-accent">{tc.tool}</span>
                    <span className="text-gray-500">
                      ({JSON.stringify(tc.args).substring(0, 60)})
                    </span>
                    <p className="text-gray-600 mt-1 truncate">
                      {tc.result.substring(0, 100)}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-gray-600">No actions yet</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
