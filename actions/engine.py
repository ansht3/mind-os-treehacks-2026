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

            elif suggestion.action_type == "find_text":
                await self.browser.find_text(detail.get("text", ""))

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

        action_result = ""  # Feedback from previous action to send to LLM
        consecutive_failures = 0

        for step in range(1, self.max_steps + 1):
            # Ensure cursor is alive after page updates
            await self.browser.ensure_cursor()

            # Get current page context
            url_before = await self.browser.get_url()
            page_title = await self.browser.get_page_title()

            try:
                self._elements = await self.browser.get_interactive_elements()
            except Exception:
                await asyncio.sleep(1)
                try:
                    self._elements = await self.browser.get_interactive_elements()
                except Exception:
                    self._elements = []

            print(f"  [{step}] {page_title} — {url_before}", flush=True)
            print(f"       {len(self._elements)} elements found", flush=True)

            # Display top 10 available actions for visibility
            self._print_suggestions(self._elements[:10])

            page_text = await self.browser.get_page_text()

            # Ask LLM for next action
            action = await self.planner.decide_next_action(
                goal=goal,
                current_url=url_before,
                page_title=page_title,
                elements=self._elements,
                step_number=step,
                page_text=page_text,
                action_result=action_result,
            )
            action_result = ""  # Reset for this step

            if action is None:
                consecutive_failures += 1
                print(f"       LLM failed to respond ({consecutive_failures}/3)", flush=True)
                if consecutive_failures >= 3:
                    print("       Too many LLM failures, stopping.\n", flush=True)
                    return
                await asyncio.sleep(1)
                continue
            consecutive_failures = 0

            # Check if done
            if action.action_type == "done":
                summary = action.action_detail.get("summary", "Goal completed")
                print(f"       DONE: {summary}\n", flush=True)
                return

            # Handle confirmation requests
            if action.action_type == "confirm":
                question = action.action_detail.get("question", "Should I proceed?")
                print(f"       ? {question}", flush=True)
                answer = input("       > ").strip()
                if answer.lower() in ("q", "quit", "exit"):
                    print("       Stopped by user.\n", flush=True)
                    return
                self.planner._messages.append({"role": "user", "content": f"User response: {answer}"})
                continue

            # Log and execute
            print(f"       Action: {action.action_type} — {action.description}", flush=True)
            action_result = await self._execute(action)
            print(f"       Result: {action_result}", flush=True)

            # Wait for page to update — longer for navigation actions
            if action.action_type in ("type", "navigate", "click"):
                await asyncio.sleep(2)
                # Extra wait if URL changed (page navigation)
                url_after = await self.browser.get_url()
                if url_after != url_before:
                    await asyncio.sleep(1)
            else:
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

    def _find_element_by_text(self, text_hint: str) -> dict | None:
        """Fuzzy fallback: find an element whose text contains the hint."""
        if not text_hint:
            return None
        hint_lower = text_hint.lower()
        for el in self._elements:
            if hint_lower in el.get("text", "").lower():
                return el
        return None

    async def _execute(self, action: Suggestion) -> str:
        """Execute a single action on the browser. Returns a result string for the LLM."""
        detail = action.action_detail
        try:
            if action.action_type == "click":
                el = self._find_element(detail.get("element_id", -1))
                if not el:
                    # Fallback: try to find by reasoning text
                    el = self._find_element_by_text(action.description)
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])
                    return f"Clicked \"{el['text'][:50]}\""
                else:
                    return f"ERROR: element_id {detail.get('element_id')} not found in current elements"

            elif action.action_type == "type":
                el = self._find_element(detail.get("element_id", -1))
                if not el:
                    # Fallback: find first input element
                    for e in self._elements:
                        if e["tag"] in ("input", "textarea") or e.get("type") in ("text", "search"):
                            el = e
                            break
                if el:
                    await self.browser.click_coords(el["cx"], el["cy"])
                    text = detail.get("text", "")
                    if text:
                        await self.browser.page_type(text)
                        # Auto-press Enter after typing (search submission)
                        await self.browser.press_key("Enter")
                        return f"Typed \"{text}\" and pressed Enter"
                    return f"Clicked input \"{el['text'][:30]}\""
                else:
                    return "ERROR: no input element found on page"

            elif action.action_type == "navigate":
                url = detail.get("url", "")
                if url:
                    await self.browser.goto(url)
                    return f"Navigated to {url[:60]}"
                return "ERROR: no URL provided"

            elif action.action_type == "scroll":
                direction = detail.get("direction", "down")
                await self.browser.scroll(direction)
                return f"Scrolled {direction}"

            elif action.action_type == "press_key":
                key = detail.get("key", "Enter")
                await self.browser.press_key(key)
                return f"Pressed {key}"

            elif action.action_type == "find_text":
                # find_text opens Cmd+F which breaks the browser state.
                # Instead, just scroll down — the LLM will see updated page text.
                await self.browser.scroll("down")
                return "Scrolled down to find content (find_text converted to scroll)"

            elif action.action_type == "history":
                if detail.get("direction") == "back":
                    await self.browser.go_back()
                    return "Went back"
                else:
                    await self.browser.go_forward()
                    return "Went forward"

            else:
                return f"Unknown action type: {action.action_type}"

        except Exception as e:
            return f"ERROR: {e}"

    async def stop(self):
        """Close the browser."""
        await self.browser.close()
