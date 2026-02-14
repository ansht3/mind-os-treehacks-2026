/**
 * Agent Orchestrator: bridges EMG commands to OpenAI Responses API + MCP.
 *
 * Receives routed commands and calls the OpenAI Responses API with:
 * - MCP tool pointing to our browser control server
 * - Constrained prompting for single-action steps
 * - Conversation chaining via previous_response_id
 */

import OpenAI from "openai";
import { AgentState, ToolCallRecord } from "./state.js";
import { SYSTEM_PROMPT, buildUserPrompt } from "./prompts.js";
import { RoutedCommand } from "./command_router.js";

const MCP_SERVER_URL = process.env.MCP_SERVER_URL || "http://localhost:3333/sse";
const AGENT_MODEL = process.env.AGENT_MODEL || "gpt-4o";

export class Orchestrator {
  private client: OpenAI;
  private state: AgentState;

  constructor(state: AgentState) {
    this.client = new OpenAI();
    this.state = state;
  }

  get agentState(): AgentState {
    return this.state;
  }

  setGoal(goal: string): void {
    this.state.goal = goal;
    this.state.plan = [];
    this.state.currentStep = 0;
    this.state.lastResponseId = null;
    this.state.toolHistory = [];
    this.state.active = true;
  }

  /**
   * Execute a routed command. For DIRECT commands, calls MCP directly.
   * For AGENT commands, goes through the LLM.
   */
  async execute(routed: RoutedCommand): Promise<{
    action: string;
    result: string;
    toolCalls: ToolCallRecord[];
  }> {
    this.state.lastCommand = routed.command;

    if (routed.mode === "DIRECT" && routed.directAction) {
      return this.executeDirect(routed);
    }

    if (routed.mode === "CONTROL") {
      return this.executeControl(routed);
    }

    return this.executeAgent(routed);
  }

  /**
   * Direct execution: call MCP tool without LLM.
   */
  private async executeDirect(routed: RoutedCommand): Promise<{
    action: string;
    result: string;
    toolCalls: ToolCallRecord[];
  }> {
    const action = routed.directAction!;
    const record: ToolCallRecord = {
      timestamp: Date.now(),
      tool: action.tool,
      args: action.args,
      result: `Direct: ${action.tool}(${JSON.stringify(action.args)})`,
    };
    this.state.toolHistory.push(record);

    return {
      action: `Direct: ${routed.command}`,
      result: record.result,
      toolCalls: [record],
    };
  }

  /**
   * Control command execution (CONFIRM/CANCEL).
   */
  private async executeControl(routed: RoutedCommand): Promise<{
    action: string;
    result: string;
    toolCalls: ToolCallRecord[];
  }> {
    if (routed.command === "CANCEL") {
      this.state.active = false;
      return {
        action: "Cancelled",
        result: "Agent task cancelled by user.",
        toolCalls: [],
      };
    }

    // CONFIRM: advance to next step
    if (this.state.currentStep < this.state.plan.length - 1) {
      this.state.currentStep++;
    }

    return {
      action: "Confirmed",
      result: `Proceeding to step ${this.state.currentStep + 1}.`,
      toolCalls: [],
    };
  }

  /**
   * Agent execution: call OpenAI Responses API with MCP tool.
   */
  private async executeAgent(routed: RoutedCommand): Promise<{
    action: string;
    result: string;
    toolCalls: ToolCallRecord[];
  }> {
    const userPrompt = buildUserPrompt(
      this.state,
      routed.command,
      undefined
    );

    try {
      const params: Record<string, unknown> = {
        model: AGENT_MODEL,
        instructions: SYSTEM_PROMPT,
        tools: [
          {
            type: "mcp",
            server_label: "browser",
            server_url: MCP_SERVER_URL,
            require_approval: "never",
          },
        ],
        input: userPrompt,
        truncation: "auto",
      };

      // Chain conversation if we have a previous response
      if (this.state.lastResponseId) {
        params.previous_response_id = this.state.lastResponseId;
      }

      const response = await this.client.responses.create(params as any);

      // Save response ID for chaining
      this.state.lastResponseId = response.id;

      // Extract tool calls and text from response
      const toolCalls: ToolCallRecord[] = [];
      let resultText = "";

      for (const item of response.output) {
        if ((item as any).type === "mcp_call") {
          const mcpCall = item as any;
          toolCalls.push({
            timestamp: Date.now(),
            tool: mcpCall.name,
            args: JSON.parse(mcpCall.arguments || "{}"),
            result: (mcpCall.output || "").substring(0, 500),
          });
        }

        if ((item as any).type === "message") {
          const msg = item as any;
          for (const content of msg.content || []) {
            if (content.type === "output_text" || content.type === "text") {
              resultText += content.text + "\n";
            }
          }
        }
      }

      // Also check output_text shorthand
      if (!resultText && (response as any).output_text) {
        resultText = (response as any).output_text;
      }

      // Update state
      this.state.toolHistory.push(...toolCalls);

      return {
        action: `Agent: ${routed.command}`,
        result: resultText || "Action executed.",
        toolCalls,
      };
    } catch (error) {
      const errMsg = error instanceof Error ? error.message : String(error);
      console.error("[Orchestrator] Error:", errMsg);

      return {
        action: `Error: ${routed.command}`,
        result: `Failed: ${errMsg}`,
        toolCalls: [],
      };
    }
  }
}
