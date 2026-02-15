"""LLM-powered agent planner for autonomous browser automation."""

import json
import re
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


SYSTEM_PROMPT = """You are a browser automation agent. You receive a goal and execute it step by step.

Return ONLY valid JSON (no markdown, no commentary):
{"action_type": "<type>", "action_detail": {<details>}, "reasoning": "brief why"}

ACTIONS:
- type: {"element_id": N, "text": "what to type"}  — clicks the element, types text, then presses Enter automatically
- click: {"element_id": N}  — clicks the element
- scroll: {"direction": "up|down"}  — scrolls the page
- navigate: {"url": "https://..."}  — go to a URL directly
- press_key: {"key": "Enter|Tab|Escape"}  — press a single key
- done: {"summary": "what was accomplished"}  — goal is fully complete

CRITICAL RULES:
1. MULTI-STEP GOALS: Break goals into steps. "Search X and add to cart" = type search → click product → click Add to Cart → done. NEVER return done until the FULL goal is met.
2. TYPE = SEARCH: When you use "type", it auto-presses Enter afterward. Do NOT follow a type with press_key Enter.
3. ADD TO CART FLOW: On shopping sites, the FASTEST path is: search → scroll to see results → click "Add to Cart" button next to cheapest item → done. You do NOT need to sort or visit the product page — many sites show "Add to Cart" buttons directly in search results.
4. ELEMENT SELECTION: Always use element_id from the CURRENT element list. IDs change each step. Pick the element whose text best matches your intent. If the element you need is not visible, scroll down.
5. VERIFY BEFORE DONE: Before returning done, check that the page state confirms your goal. For "add to cart", you should see cart confirmation (e.g., "Added to Cart", "Go to Cart", or cart count increased).
6. POPUPS/OVERLAYS: Dismiss sign-in prompts, cookie banners, or popups by pressing Escape or clicking their close/dismiss button. Then continue with the goal.
7. CAPTCHA/BOT BLOCK: If you see a CAPTCHA or bot-check, return done with summary explaining the block.
8. NEVER REPEAT A FAILED ACTION: If the Result of your last action says ERROR or the page didn't change, you MUST try something different. Scroll down, click a different element, or take an alternative approach. NEVER click the same element twice in a row.
9. ONE ACTION PER STEP: Return exactly one action. Never try to combine multiple actions.
10. SKIP SORTING: Do NOT try to sort search results — sort dropdowns are unreliable. Instead, just scroll through results and pick the cheapest item you can see from the prices listed."""


class AgentPlanner:
    """Decides the next action to take given goal, page state, and history."""

    def __init__(self, api_key: str = OPENAI_API_KEY):
        self._client = AsyncOpenAI(api_key=api_key)
        self._messages: list[dict] = []
        self._last_url: str = ""
        self._last_action: str = ""
        self._stuck_count: int = 0

    def reset(self):
        """Clear conversation history for a new goal."""
        self._messages = []
        self._last_url = ""
        self._last_action = ""
        self._stuck_count = 0

    async def decide_next_action(
        self,
        goal: str,
        current_url: str,
        page_title: str,
        elements: list[dict],
        step_number: int,
        page_text: str = "",
        action_result: str = "",
    ) -> Suggestion | None:
        """Decide the next action based on current page state.

        Returns a Suggestion to execute, or a Suggestion with action_type="done"
        when the goal is complete. Returns None on LLM failure.
        """
        # Detect if we're stuck (same URL and repeated action)
        if current_url == self._last_url and self._stuck_count >= 1:
            stuck_hint = f"\nWARNING: You have been on this same page for {self._stuck_count + 1} steps. Your last action did NOT change the page. You MUST try a DIFFERENT action. Do NOT repeat what you just did."
        else:
            stuck_hint = ""
        if current_url != self._last_url:
            self._stuck_count = 0
        else:
            self._stuck_count += 1
        self._last_url = current_url

        # Build compact element list
        el_parts = []
        for el in elements[:60]:
            p = f"[{el['id']}] {el['tag']}: \"{el['text']}\""
            if el.get("href"):
                p += f" href={el['href'][:80]}"
            if el.get("type"):
                p += f" type={el['type']}"
            el_parts.append(p)
        el_text = "\n".join(el_parts) if el_parts else "(no interactive elements found)"

        # Include page text — skip the first 200 chars (usually nav) and take more content
        page_snippet = ""
        if page_text:
            # Take a meaningful slice of the page content
            useful_text = page_text[200:4000].strip()
            if useful_text:
                page_snippet = f"\nPage content (truncated):\n{useful_text}\n"

        # Build result feedback from previous action
        result_note = ""
        if action_result:
            result_note = f"\nResult of last action: {action_result}"
            if "ERROR" in action_result:
                result_note += "\n^^^ The last action FAILED. You must adjust your approach."
            result_note += "\n"

        user_content = f"""Goal: {goal}

Step: {step_number}
URL: {current_url}
Title: {page_title}
{result_note}{page_snippet}{stuck_hint}
Interactive elements:
{el_text}

What is the single next action?"""

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

            # Trim history to keep context manageable
            # Keep system prompt + last 14 turns (7 exchanges)
            if len(self._messages) > 29:
                self._messages = [self._messages[0]] + self._messages[-28:]

            action = _parse_llm_json(raw)
        except Exception as e:
            # On failure, remove the user message we just appended so we don't
            # poison the conversation with an unanswered message
            if self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
            print(f"       LLM error: {e}", flush=True)
            return None

        if action is None:
            # JSON parse failed — remove the bad assistant message too
            if self._messages and self._messages[-1]["role"] == "assistant":
                self._messages.pop()
            if self._messages and self._messages[-1]["role"] == "user":
                self._messages.pop()
            return None

        action_type = action.get("action_type", "done")
        action_detail = action.get("action_detail", {})
        reasoning = action.get("reasoning", "")

        self._last_action = action_type

        return Suggestion(
            id=0,
            label=reasoning,
            action_type=action_type,
            action_detail=action_detail,
            description=reasoning,
        )


def _parse_llm_json(raw: str) -> dict | None:
    """Robustly parse JSON from LLM output, handling common formatting issues."""
    # Strip markdown code fences
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
        text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from surrounding text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try fixing common issues: trailing commas, single quotes
    cleaned = text.replace("'", '"')
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    return None
