"""Autonomous browser agent — takes a goal and executes it step by step."""

import asyncio

from actions.browser import BrowserController
from actions.vision import AgentPlanner, Suggestion
from actions.config import OPENAI_API_KEY, MAX_AGENT_STEPS


class AutonomousAgent:
    """Takes a natural language goal and autonomously controls the browser to achieve it."""

    def __init__(self, max_steps: int = MAX_AGENT_STEPS):
        self.browser = BrowserController()
        self.planner = AgentPlanner(api_key=OPENAI_API_KEY)
        self.max_steps = max_steps
        self._elements: list[dict] = []

    async def start(self, url: str):
        """Launch browser and navigate to starting URL."""
        await self.browser.launch()
        await self.browser.goto(url)

    async def run(self, goal: str):
        """Run the autonomous agent loop until goal is done or max steps reached."""
        self.planner.reset()
        print(f"\n  Goal: {goal}\n", flush=True)

        for step in range(1, self.max_steps + 1):
            # Ensure cursor is alive after page updates
            await self.browser.ensure_cursor()

            # Get current page context
            current_url = await self.browser.get_url()
            page_title = await self.browser.get_page_title()

            try:
                self._elements = await self.browser.get_interactive_elements()
            except Exception:
                await asyncio.sleep(0.5)
                try:
                    self._elements = await self.browser.get_interactive_elements()
                except Exception:
                    self._elements = []

            print(f"  [{step}] {page_title} — {current_url}", flush=True)
            print(f"       {len(self._elements)} elements found", flush=True)

            # Display top 10 available actions for visibility
            self._print_suggestions(self._elements[:10])

            # Ask LLM for next action
            action = await self.planner.decide_next_action(
                goal=goal,
                current_url=current_url,
                page_title=page_title,
                elements=self._elements,
                step_number=step,
            )

            if action is None:
                print(f"       LLM failed to respond, retrying...", flush=True)
                await asyncio.sleep(1)
                continue

            # Check if done
            if action.action_type == "done":
                summary = action.action_detail.get("summary", "Goal completed")
                print(f"       DONE: {summary}\n", flush=True)
                return

            # Log and execute
            print(f"       Action: {action.action_type} — {action.description}", flush=True)
            await self._execute(action)

            # Restore cursor after action (navigation may have wiped it)
            await self.browser.ensure_cursor()

            # Brief pause for page to update
            await asyncio.sleep(1)

        print(f"\n  Reached max steps ({self.max_steps}). Stopping.\n", flush=True)

    def _print_suggestions(self, elements: list[dict]):
        """Print the top interactive elements as numbered suggestions."""
        if not elements:
            print("       (no interactive elements)", flush=True)
            return
        print("       ┌─ Available actions ─────────────────────", flush=True)
        for i, el in enumerate(elements):
            tag = el.get("tag", "?")
            text = el.get("text", "")[:45]
            href = el.get("href", "")
            suffix = f" → {href[:40]}" if href else ""
            print(f"       │ [{i}] <{tag}> {text}{suffix}", flush=True)
        print("       └────────────────────────────────────────", flush=True)

    def _find_element(self, element_id: int) -> dict | None:
        for el in self._elements:
            if el["id"] == element_id:
                return el
        return None

    async def _execute(self, action: Suggestion):
        """Execute a single action on the browser."""
        detail = action.action_detail
        try:
            if action.action_type == "click":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])

            elif action.action_type == "type":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])
                text = detail.get("text", "")
                if text:
                    await self.browser.page_type(text)

            elif action.action_type == "navigate":
                url = detail.get("url", "")
                if url:
                    await self.browser.goto(url)

            elif action.action_type == "scroll":
                await self.browser.scroll(detail.get("direction", "down"))

            elif action.action_type == "press_key":
                await self.browser.press_key(detail.get("key", "Enter"))

            elif action.action_type == "history":
                if detail.get("direction") == "back":
                    await self.browser.go_back()
                else:
                    await self.browser.go_forward()

        except Exception as e:
            print(f"       Action failed: {e}", flush=True)

    async def stop(self):
        """Close the browser."""
        await self.browser.close()
