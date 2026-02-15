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


SYSTEM_PROMPT = """You are a focused browser automation agent. You ONLY do exactly what the user asked — nothing more.

Given:
- The user's goal
- Current page URL and title
- Interactive elements on the page (with IDs, text, types, positions)
- History of actions you've already taken

Decide the SINGLE next action. Return ONLY a JSON object (no markdown, no extra text):
{"action_type": "click|type|scroll|navigate|press_key|find_text|confirm|done", "action_detail": {}, "reasoning": "brief explanation"}

action_detail formats:
- click: {"element_id": N}
- type: {"element_id": N, "text": "what to type"}  (clicks the element first, then types)
- scroll: {"direction": "up|down"}
- navigate: {"url": "https://..."}
- press_key: {"key": "Enter|Tab|Escape|..."}
- find_text: {"text": "search term"}  — opens browser Find (Cmd+F) and searches for text on the current page. Use this to locate specific info on long pages (e.g. Wikipedia).
- confirm: {"question": "Should I do X?"}  — ASK the user before proceeding
- done: {"summary": "what was accomplished"}

CRITICAL RULES:
1. ONLY take actions that directly accomplish the user's stated goal. Do NOT interact with popups, banners, language selectors, cookie prompts, sign-in prompts, or anything unrelated to the goal — just ignore them.
2. ALWAYS use "confirm" before clicking a link that navigates to a new page (e.g. a search result, a product link, an article). The user must approve which result to visit. Example: {"action_type": "confirm", "action_detail": {"question": "Click on 'Shawn Shen - LinkedIn'?"}}
3. If you are unsure whether an action is what the user wants, use "confirm" to ask them first. Examples: choosing between products, selecting options the user didn't specify, clicking something that might change account settings.
4. NEVER change settings (language, location, preferences, account details) unless the user explicitly asked.
5. Take the shortest path to the goal. Skip unnecessary steps.
6. Use element_id from the provided element list — pick the best match.
7. For search: type into the search box, then press_key Enter in the next step. The field will be cleared automatically before typing — do NOT click or clear it yourself.
8. Return "done" when the goal is accomplished or truly cannot be completed.
9. PAY CLOSE ATTENTION TO USER RESPONSES. When the user says "no" to a confirm and provides extra info (e.g. "no, he's from Georgia Tech"), incorporate that info into your next action. If the current search results don't match, REFINE your search query with the new details (e.g. search "Shawn Shen Georgia Tech" instead of "Shawn Shen" again). Never ignore hints the user gives you.
10. NEVER repeat the same search query. If you already searched for something, DO NOT type it again. Look at the results already on the page. If you need different results, use a DIFFERENT, more specific query.
11. READ THE PAGE before acting. If search results are visible, examine them — do not re-search. If none match the goal, scroll down to see more results OR refine your search with additional keywords.
12. NEVER go back to re-do something you already did. Move forward.
13. If you get stuck on the same page, try scrolling or a different approach — do NOT click random UI elements.
14. NEVER use "history" (back/forward) more than once in a row. If going back didn't help, try navigating directly to a URL instead.
15. after every one click should be another question to user. user needs to accept before making another action.
16. IMPORTANT — FINDING INFO ON A PAGE: When you are on a content-heavy page (Wikipedia, articles, docs) and need to find specific info (a date, a name, a fact), you MUST use "find_text" IMMEDIATELY — do NOT scroll. For example, to find Ronaldo's birthday on Wikipedia, use {"action_type": "find_text", "action_detail": {"text": "Born"}}. NEVER scroll more than twice on the same page looking for information — use find_text instead.
17. If you have already scrolled twice on the same page without finding what you need, your next action MUST be find_text."""


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
