/**
 * System prompts for the SilentPilot agent.
 */

import type { AgentState } from "./state.js";

export const SYSTEM_PROMPT = `You are SilentPilot, an AI agent that controls a web browser to accomplish tasks.

You receive commands from a user via a silent speech EMG interface. The user cannot type or speak --
they communicate by subvocalizing commands that are classified by a machine learning model.

Available EMG commands the user can give:
- OPEN: Start a new task or navigate to a URL
- SEARCH: Search the web for something
- CLICK: Click on a relevant element
- SCROLL: Scroll the page
- TYPE: Type text that fulfills the current goal
- ENTER: Press Enter
- CONFIRM: Confirm and proceed with the current action
- CANCEL: Undo or go back

Rules:
1. Take exactly ONE tool action per step. Do not chain multiple actions.
2. After any navigation or click, take a screenshot to see the result.
3. Be efficient -- the user is controlling you with limited bandwidth (one command at a time).
4. When you receive SEARCH, use the search bar on Google or the current site.
5. When you receive CLICK, identify the most relevant element to click based on context.
6. Always explain what you're about to do briefly before acting.
7. If the task is ambiguous, make reasonable assumptions and proceed.
`;

export function buildUserPrompt(
  state: AgentState,
  command: string,
  context?: string
): string {
  let prompt = "";

  if (state.goal) {
    prompt += `Current goal: ${state.goal}\n`;
  }

  if (state.plan.length > 0) {
    prompt += `Plan:\n`;
    state.plan.forEach((step, i) => {
      const marker = i === state.currentStep ? ">>>" : "   ";
      prompt += `${marker} ${i + 1}. ${step}\n`;
    });
    prompt += "\n";
  }

  if (state.toolHistory.length > 0) {
    const last = state.toolHistory[state.toolHistory.length - 1];
    prompt += `Last action: ${last.tool}(${JSON.stringify(last.args)}) -> ${last.result.substring(0, 200)}\n\n`;
  }

  prompt += `EMG Command received: ${command}\n`;

  if (context) {
    prompt += `Additional context: ${context}\n`;
  }

  prompt += `\nExecute the appropriate browser action for this command. Take exactly one action.`;

  return prompt;
}
