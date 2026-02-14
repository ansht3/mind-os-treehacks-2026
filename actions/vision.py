"""OpenAI Vision API integration for analyzing screenshots and suggesting actions."""

import base64
import json
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from actions.config import OPENAI_API_KEY, OPENAI_MODEL, NUM_SUGGESTIONS


@dataclass
class Suggestion:
    """A single suggested browser action."""

    id: int
    label: str
    action_type: str  # click, scroll, type, navigate, press_key
    action_detail: dict = field(default_factory=dict)
    description: str = ""


SYSTEM_PROMPT = f"""\
You are a browser automation assistant. You receive a screenshot, the current URL, and a list of interactive elements detected on the page (with their IDs, text, and positions).

Return exactly {NUM_SUGGESTIONS} suggested next actions as a JSON array (no markdown, no code fences).

Each element has:
- "id": integer 0-{NUM_SUGGESTIONS - 1}
- "label": short human-readable label (max 60 chars)
- "action_type": one of "click", "scroll", "type", "navigate", "press_key"
- "action_detail": object depending on action_type:
  - click: {{"element_id": N}} — the ID from the interactive elements list
  - scroll: {{"direction": "up" or "down"}}
  - type: {{"element_id": N, "text": "text to type"}} — click the element then type
  - navigate: {{"url": "full URL"}}
  - press_key: {{"key": "Enter" or other key name}}
- "description": brief explanation of what this action does

IMPORTANT:
- For click and type actions, use "element_id" referencing an element from the provided list.
- Prioritize the most useful/common actions first.
- Always include at least one scroll option.
- If an element has an href, you may use "navigate" with that URL instead of clicking."""


class PageAnalyzer:
    """Sends screenshots + element list to OpenAI Vision API."""

    def __init__(self, api_key: str = OPENAI_API_KEY):
        self._client = AsyncOpenAI(api_key=api_key)

    async def analyze(
        self,
        screenshot_bytes: bytes,
        current_url: str,
        elements: list[dict],
    ) -> list[Suggestion]:
        """Analyze a screenshot + element list and return suggested actions."""
        b64_image = base64.b64encode(screenshot_bytes).decode("utf-8")

        # Build a compact element list string
        el_lines = []
        for el in elements:
            line = f"[{el['id']}] <{el['tag']}> \"{el['text']}\""
            if el.get("href"):
                line += f" href={el['href']}"
            if el.get("type"):
                line += f" type={el['type']}"
            el_lines.append(line)
        elements_text = "\n".join(el_lines) if el_lines else "(no interactive elements found)"

        response = await self._client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Current URL: {current_url}\n\nInteractive elements on page:\n{elements_text}\n\nSuggest {NUM_SUGGESTIONS} actions.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64_image}",
                                "detail": "low",
                            },
                        },
                    ],
                },
            ],
            max_tokens=2000,
        )

        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            raw = raw.rsplit("```", 1)[0]

        try:
            items = json.loads(raw)
        except json.JSONDecodeError:
            return [
                Suggestion(
                    id=0,
                    label="Scroll down",
                    action_type="scroll",
                    action_detail={"direction": "down"},
                    description="Scroll down the page (fallback)",
                )
            ]

        suggestions = []
        for item in items[:NUM_SUGGESTIONS]:
            suggestions.append(
                Suggestion(
                    id=item.get("id", len(suggestions)),
                    label=item.get("label", "Unknown"),
                    action_type=item.get("action_type", "scroll"),
                    action_detail=item.get("action_detail", {}),
                    description=item.get("description", ""),
                )
            )
        return suggestions
