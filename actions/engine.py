"""Browser automation engines — GUI interactive mode and autonomous agent mode."""

import asyncio

from actions.browser import BrowserController
from actions.vision import AgentPlanner, Suggestion
from actions.overlay import Overlay
from actions.config import OPENAI_API_KEY, MAX_AGENT_STEPS


class ActionEngine:
    """Interactive mode: shows DOM elements as suggestions, user picks one."""

    def __init__(self, overlay=None):
        self.browser = BrowserController(on_cursor_move=self._on_cursor_position)
        self.overlay = overlay if overlay is not None else Overlay()
        self._suggestions: list[Suggestion] = []
        self._elements: list[dict] = []
        self._selected: int = 0
        self._running: bool = False

    def _on_cursor_position(self, x, y, action):
        if hasattr(self.overlay, 'update_cursor_info'):
            self.overlay.update_cursor_info(x, y, action)

    async def start(self, url: str):
        self.overlay.set_status(False)
        await self.browser.launch()
        await self.browser.goto(url)
        self._running = True

    def _elements_to_suggestions(self, elements: list[dict]) -> list[Suggestion]:
        """Convert raw DOM elements into suggestions."""
        suggestions = []
        for el in elements[:8]:
            if el.get("href"):
                action_type = "navigate"
                detail = {"url": el["href"], "element_id": el["id"]}
            elif (
                el["tag"] in ("input", "textarea", "select")
                or el.get("type") in ("text", "search", "email", "password", "url")
                or "field)" in el.get("text", "").lower()
                or "search" in el.get("text", "").lower()
            ):
                action_type = "type"
                detail = {"element_id": el["id"], "text": ""}
            else:
                action_type = "click"
                detail = {"element_id": el["id"]}

            suggestions.append(Suggestion(
                id=len(suggestions),
                label=el["text"][:50],
                action_type=action_type,
                action_detail=detail,
                description=f"{action_type} <{el['tag']}>",
            ))

        suggestions.append(Suggestion(
            id=len(suggestions), label="Type / Search", action_type="type_anywhere",
            action_detail={"text": ""}, description="Type into search or input",
        ))
        suggestions.append(Suggestion(
            id=len(suggestions), label="Scroll Down", action_type="scroll",
            action_detail={"direction": "down"}, description="Scroll down",
        ))
        suggestions.append(Suggestion(
            id=len(suggestions), label="Scroll Up", action_type="scroll",
            action_detail={"direction": "up"}, description="Scroll up",
        ))
        suggestions.append(Suggestion(
            id=len(suggestions), label="Go Back", action_type="history",
            action_detail={"direction": "back"}, description="Previous page",
        ))
        suggestions.append(Suggestion(
            id=len(suggestions), label="Go Forward", action_type="history",
            action_detail={"direction": "forward"}, description="Next page",
        ))
        return suggestions

    async def run_cycle(self) -> list[Suggestion]:
        """Extract DOM elements and show as suggestions."""
        self.overlay.set_status(False)
        await self.browser.ensure_cursor()

        current_url = await self.browser.get_url()
        try:
            self._elements = await self.browser.get_interactive_elements()
        except Exception:
            await asyncio.sleep(0.5)
            self._elements = await self.browser.get_interactive_elements()
        self._suggestions = self._elements_to_suggestions(self._elements)
        self._selected = 0

        self.overlay.set_status(True)
        self.overlay.show(self._suggestions, self._selected, smart=False)
        return self._suggestions

    def move_selection(self, direction: str):
        if not self._suggestions:
            return
        if direction == "up":
            self._selected = max(0, self._selected - 1)
        else:
            self._selected = min(len(self._suggestions) - 1, self._selected + 1)
        self.overlay.show(self._suggestions, self._selected)

    def select_index(self, index: int):
        if self._suggestions and 0 <= index < len(self._suggestions):
            self._selected = index
            self.overlay.show(self._suggestions, self._selected)

    def get_selected(self) -> "Suggestion | None":
        if self._suggestions and 0 <= self._selected < len(self._suggestions):
            return self._suggestions[self._selected]
        return None

    def set_type_text(self, text: str):
        """Set the text for the currently selected type action."""
        s = self.get_selected()
        if s and s.action_type in ("type", "type_anywhere"):
            s.action_detail["text"] = text

    async def execute_selected(self) -> bool:
        """Execute the selected suggestion. Returns True if a text field was clicked
        (so the caller can prompt for typing)."""
        if not self._suggestions:
            return False
        self.overlay.set_status(False)
        suggestion = self._suggestions[self._selected]
        await self.execute(suggestion)
        # After a click, check if we landed on a text field
        if suggestion.action_type == "click":
            try:
                if await self.browser.is_focused_typeable():
                    return True
            except Exception:
                pass
        return False

    def _find_element(self, element_id: int) -> dict | None:
        for el in self._elements:
            if el["id"] == element_id:
                return el
        return None

    async def execute(self, suggestion: Suggestion):
        detail = suggestion.action_detail
        try:
            if suggestion.action_type == "click":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])

            elif suggestion.action_type == "type":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])
                text = detail.get("text", "")
                if text:
                    await self.browser.page_type(text)
                    await self.browser.press_key("Enter")

            elif suggestion.action_type == "navigate":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])
                else:
                    await self.browser.goto(detail.get("url", ""))

            elif suggestion.action_type == "type_anywhere":
                await self.browser.focus_and_type(detail.get("text", ""))

            elif suggestion.action_type == "scroll":
                await self.browser.scroll(detail.get("direction", "down"))

            elif suggestion.action_type == "press_key":
                await self.browser.press_key(detail.get("key", "Enter"))

            elif suggestion.action_type == "history":
                if detail.get("direction") == "back":
                    await self.browser.go_back()
                else:
                    await self.browser.go_forward()

        except Exception as e:
            print(f"\n  Action failed: {e}", flush=True)

    async def stop(self):
        self._running = False
        await self.browser.close()
        self.overlay.clear()


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
