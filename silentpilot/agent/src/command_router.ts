/**
 * Command Router: maps EMG CommandEvents to either direct actions or agent intents.
 *
 * "Direct" commands bypass the LLM (faster, deterministic):
 *   SCROLL, ENTER -> executed immediately via the MCP server
 *
 * "Agent" commands go through the orchestrator (need LLM reasoning):
 *   OPEN, SEARCH, CLICK, TYPE -> need context to decide what to do
 *
 * "Control" commands manage the agent flow:
 *   CONFIRM, CANCEL -> commit or undo
 */

export type CommandMode = "DIRECT" | "AGENT" | "CONTROL";

export interface CommandEvent {
  cmd: string;
  confidence: number;
  mode: CommandMode;
  context?: Record<string, unknown>;
}

export interface RoutedCommand {
  command: string;
  mode: CommandMode;
  /** For direct commands: the specific action to take */
  directAction?: {
    tool: string;
    args: Record<string, unknown>;
  };
}

/** Commands that bypass the LLM */
const DIRECT_COMMANDS: Record<string, { tool: string; args: Record<string, unknown> }> = {
  SCROLL: { tool: "browser_scroll", args: { direction: "down", amount: 300 } },
  ENTER: { tool: "browser_press", args: { key: "Enter" } },
};

/** Commands that need LLM reasoning */
const AGENT_COMMANDS = new Set(["OPEN", "SEARCH", "CLICK", "TYPE"]);

/** Flow control commands */
const CONTROL_COMMANDS = new Set(["CONFIRM", "CANCEL"]);

export function routeCommand(event: CommandEvent): RoutedCommand {
  const cmd = event.cmd.toUpperCase();

  // Direct execution
  if (cmd in DIRECT_COMMANDS) {
    return {
      command: cmd,
      mode: "DIRECT",
      directAction: DIRECT_COMMANDS[cmd],
    };
  }

  // Agent reasoning needed
  if (AGENT_COMMANDS.has(cmd)) {
    return {
      command: cmd,
      mode: "AGENT",
    };
  }

  // Flow control
  if (CONTROL_COMMANDS.has(cmd)) {
    return {
      command: cmd,
      mode: "CONTROL",
    };
  }

  // Unknown command -> agent
  return {
    command: cmd,
    mode: "AGENT",
  };
}
