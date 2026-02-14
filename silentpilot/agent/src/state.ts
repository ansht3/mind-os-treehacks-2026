/**
 * Shared agent state across the orchestrator lifecycle.
 */

export interface AgentState {
  /** Current high-level goal set by the user/judge */
  goal: string;

  /** Current step plan (list of steps to accomplish the goal) */
  plan: string[];

  /** Index of the current step being executed */
  currentStep: number;

  /** Last OpenAI response ID for conversation chaining */
  lastResponseId: string | null;

  /** History of tool calls made */
  toolHistory: ToolCallRecord[];

  /** Last EMG command received */
  lastCommand: string | null;

  /** Whether the agent is actively executing */
  active: boolean;
}

export interface ToolCallRecord {
  timestamp: number;
  tool: string;
  args: Record<string, unknown>;
  result: string;
}

export function createInitialState(goal: string = ""): AgentState {
  return {
    goal,
    plan: [],
    currentStep: 0,
    lastResponseId: null,
    toolHistory: [],
    lastCommand: null,
    active: false,
  };
}
