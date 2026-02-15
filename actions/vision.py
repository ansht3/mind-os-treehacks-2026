"""LLM-powered agent planner for autonomous browser automation."""

import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from actions.config import OPENAI_API_KEY, OPENAI_MODEL


@dataclass
class Suggestion:
    """A single browser action."""

    id: int
    label: str
    action_type: str  # click, scroll, type, navigate, press_key, done
    action_detail: dict = field(default_factory=dict)
    description: str = ""


SYSTEM_PROMPT = """You are a minimal browser agent. Do EXACTLY what the goal says — no more. Be conservative.

Return ONLY JSON (no markdown): {"action_type": "click|type|scroll|navigate|press_key|find_text|done", "action_detail": {}, "reasoning": "brief"}

action_detail:
- click: {"element_id": N}
- type: {"element_id": N, "text": "what to type"}
- scroll: {"direction": "up|down"}
- navigate: {"url": "https://..."}
- press_key: {"key": "Enter|Tab|Escape|..."}
- find_text: {"text": "search term"}
- done: {"summary": "what was accomplished"}

CRITICAL — BE CONSERVATIVE:
1. "Search X" = type X, Enter, DONE. Do NOT click results, browse, or scroll.
2. "Sort by X" or "sort by lowest" = click the sort option, DONE. Do NOT scroll. Do NOT browse.
3. Do NOT scroll unless the goal explicitly says "scroll". After search/sort/filter, STOP.
4. Return "done" immediately after the literal goal. One action (search, sort, filter) = do it + done.
5. Ignore popups, cookie prompts. Use element_id from the element list."""


class AgentPlanner:
    """Decides the next action to take given goal, page state, and history."""

    def __init__(self, api_key: str = OPENAI_API_KEY):
        self._client = AsyncOpenAI(api_key=api_key)
        self._messages: list[dict] = []

    def reset(self):
        """Clear conversation history for a new goal."""
        self._messages = []

    async def decide_next_action(
        self,
        goal: str,
        current_url: str,
        page_title: str,
        elements: list[dict],
        step_number: int,
        page_text: str = "",
    ) -> Suggestion | None:
        """Decide the next action based on current page state.

        Returns a Suggestion to execute, or a Suggestion with action_type="done"
        when the goal is complete. Returns None on LLM failure.
        """
        # Build compact element list
        el_parts = []
        for el in elements[:40]:
            p = f"[{el['id']}] {el['tag']}: \"{el['text']}\""
            if el.get("href"):
                p += f" href={el['href']}"
            if el.get("type"):
                p += f" type={el['type']}"
            el_parts.append(p)
        el_text = "\n".join(el_parts) if el_parts else "(no interactive elements found)"

        # Include truncated page text so the agent can read the page
        page_snippet = ""
        if page_text:
            page_snippet = f"\nVisible page text (truncated):\n{page_text[:2000]}\n"

        user_content = f"""Goal: {goal}

Step: {step_number}
URL: {current_url}
Page Title: {page_title}
{page_snippet}
Interactive elements:
{el_text}

What is the single next action? Remember: if you need to find specific info on this page, use find_text instead of scrolling."""

        # Build messages — system + conversation history + current state
        if not self._messages:
            self._messages.append({"role": "system", "content": SYSTEM_PROMPT})

        self._messages.append({"role": "user", "content": user_content})

        try:
            response = await self._client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=self._messages,
                max_tokens=300,
                temperature=0,
            )

            raw = response.choices[0].message.content.strip()

            # Save assistant response to history
            self._messages.append({"role": "assistant", "content": raw})

            # Trim history to avoid context overflow (keep system + last 20 turns)
            if len(self._messages) > 41:
                self._messages = [self._messages[0]] + self._messages[-40:]

            # Parse JSON
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1]
                raw = raw.rsplit("```", 1)[0]

            action = json.loads(raw)
        except (json.JSONDecodeError, Exception):
            return None

        action_type = action.get("action_type", "done")
        action_detail = action.get("action_detail", {})
        reasoning = action.get("reasoning", "")

        return Suggestion(
            id=0,
            label=reasoning,
            action_type=action_type,
            action_detail=action_detail,
            description=reasoning,
        )
