"""Main action engine — instant DOM elements + background GPT suggestions."""

import asyncio
import base64

from actions.browser import BrowserController
from actions.vision import PageAnalyzer, Suggestion
from actions.overlay import Overlay
from actions.config import OPENAI_API_KEY
from actions.cua import CUAgent
from actions.openclaw_predictor import OpenClawPredictor


class ActionEngine:
    """Shows DOM elements instantly, then upgrades to GPT suggestions in background."""

    def __init__(self, overlay=None):
        self.browser = BrowserController()
        self.analyzer = PageAnalyzer(api_key=OPENAI_API_KEY)
        self.overlay = overlay if overlay is not None else Overlay()
        self._suggestions: list[Suggestion] = []
        self._elements: list[dict] = []
        self._selected: int = 0
        self._running: bool = False
        self._gpt_task: asyncio.Task | None = None
        # Prediction mode state
        self._predictor = OpenClawPredictor()
        self._current_prediction: dict | None = None
        self._past_actions: list[str] = []
        self._rejected_actions: list[str] = []

    async def start(self, url: str):
        self.overlay.set_status(False)
        await self.browser.launch()
        await self.browser.goto(url)
        self._running = True

    def _elements_to_suggestions(self, elements: list[dict]) -> list[Suggestion]:
        """Convert raw DOM elements into suggestions (instant, no API)."""
        suggestions = []
        for el in elements[:8]:
            if el.get("href"):
                action_type = "navigate"
                detail = {"url": el["href"], "element_id": el["id"]}
            elif el["tag"] in ("input", "textarea", "select") or el.get("type") in ("text", "search", "email", "password", "url"):
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

    async def _fetch_gpt_suggestions(self, url: str, elements: list[dict]):
        """Background task: get GPT suggestions and update display."""
        try:
            gpt_suggestions = await self.analyzer.analyze(url, elements)
            if gpt_suggestions and self._running:
                self._suggestions = gpt_suggestions
                self._selected = min(self._selected, len(self._suggestions) - 1)
                self.overlay.show(self._suggestions, self._selected, smart=True)
        except Exception:
            pass  # Keep showing DOM elements if GPT fails

    async def run_cycle(self) -> list[Suggestion]:
        """Show DOM elements instantly, then upgrade with GPT in background."""
        # Cancel any pending GPT task
        if self._gpt_task and not self._gpt_task.done():
            self._gpt_task.cancel()

        # RED — extracting elements
        self.overlay.set_status(False)

        # Step 1: Instant — extract elements and show immediately
        current_url = await self.browser.get_url()
        self._elements = await self.browser.get_interactive_elements()
        self._suggestions = self._elements_to_suggestions(self._elements)
        self._selected = 0

        # GREEN — user can interact now
        self.overlay.set_status(True)
        self.overlay.show(self._suggestions, self._selected, smart=False)

        # Step 2: Background — kick off GPT for smarter suggestions
        self._gpt_task = asyncio.ensure_future(
            self._fetch_gpt_suggestions(current_url, self._elements)
        )

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

    async def execute_selected(self):
        if not self._suggestions:
            return
        # Cancel GPT if still running — we're moving on
        if self._gpt_task and not self._gpt_task.done():
            self._gpt_task.cancel()

        # RED — executing action
        self.overlay.set_status(False)

        suggestion = self._suggestions[self._selected]
        await self.execute(suggestion)

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

            elif suggestion.action_type == "navigate":
                await self.browser.goto(detail.get("url", ""))

            elif suggestion.action_type == "scroll":
                await self.browser.scroll(detail.get("direction", "down"))

            elif suggestion.action_type == "press_key":
                await self.browser.press_key(detail["key"])

            elif suggestion.action_type == "history":
                if detail.get("direction") == "back":
                    await self.browser.go_back()
                else:
                    await self.browser.go_forward()

        except Exception as e:
            print(f"\n  Action failed: {e}", flush=True)

    async def run_cua_task(self, task: str) -> str:
        """Run an autonomous CUA task on the current page.

        Args:
            task: Natural language description of what to do.

        Returns:
            Result message from the CUA agent.
        """
        # Cancel any pending GPT background task
        if self._gpt_task and not self._gpt_task.done():
            self._gpt_task.cancel()

        # RED — AI is working
        self.overlay.set_status(False)

        def _on_status(text: str):
            """Forward CUA status to overlay (best-effort)."""
            try:
                self.overlay.set_status(False)
            except Exception:
                pass

        agent = CUAgent(api_key=OPENAI_API_KEY)
        try:
            result = await agent.run(task, self.browser, on_status=_on_status)
        except Exception as e:
            result = f"CUA task failed: {e}"
            print(f"\n  {result}", flush=True)

        # GREEN — done, refresh suggestions for new page state
        self.overlay.set_status(True)
        await self.run_cycle()
        return result

    async def run_prediction_cycle(self) -> dict:
        """OpenClaw prediction cycle: predict one action, show yes/no prompt."""
        self.overlay.set_status(False)

        # Take screenshot and get page info
        screenshot_bytes = await self.browser.screenshot()
        screenshot_b64 = base64.b64encode(screenshot_bytes).decode("ascii")
        url = await self.browser.get_url()
        try:
            title = await self.browser._page.evaluate("() => document.title") if self.browser._page else url
        except Exception:
            title = url

        # Get prediction (or alternative if we have rejections)
        if self._rejected_actions:
            prediction = await self._predictor.predict_alternative(
                screenshot_b64, url, title, self._rejected_actions,
            )
        else:
            prediction = await self._predictor.predict(
                screenshot_b64, url, title, self._past_actions,
            )

        self._current_prediction = prediction
        label = prediction.get("label", "Unknown action")

        # Show prediction in overlay
        self.overlay.set_status(True)
        if hasattr(self.overlay, "show_prediction"):
            self.overlay.show_prediction(label)

        return prediction

    async def accept_prediction(self):
        """User accepted the prediction — execute it."""
        if not self._current_prediction:
            return

        self.overlay.set_status(False)
        prediction = self._current_prediction

        # Record the action
        self._past_actions.append(prediction.get("label", ""))
        self._rejected_actions.clear()

        await self.execute_prediction(prediction)
        self._current_prediction = None

    async def reject_prediction(self):
        """User rejected the prediction — record it for next alternative."""
        if not self._current_prediction:
            return
        self._rejected_actions.append(self._current_prediction.get("label", ""))
        self._current_prediction = None

    async def execute_prediction(self, prediction: dict):
        """Execute an OpenClaw prediction dict."""
        action_type = prediction.get("action_type", "")
        try:
            if action_type == "click":
                x = prediction.get("x", 0)
                y = prediction.get("y", 0)
                if x and y:
                    await self.browser.click_coords(x, y)

            elif action_type == "type":
                x = prediction.get("x", 0)
                y = prediction.get("y", 0)
                if x and y:
                    await self.browser.click_coords(x, y)
                text = prediction.get("text", "")
                if text:
                    await self.browser.page_type(text)

            elif action_type == "scroll":
                direction = prediction.get("direction", "down")
                await self.browser.scroll(direction)

            elif action_type == "navigate":
                url = prediction.get("url", "")
                if url:
                    await self.browser.goto(url)

            elif action_type == "press_key":
                key = prediction.get("key", "")
                if key:
                    await self.browser.press_key(key)

        except Exception as e:
            print(f"\n  Action failed: {e}", flush=True)

    async def stop(self):
        self._running = False
        if self._gpt_task and not self._gpt_task.done():
            self._gpt_task.cancel()
        await self.browser.close()
        self.overlay.clear()
