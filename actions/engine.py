"""Main action engine that wires browser, vision, and overlay together."""

import asyncio
import struct

from actions.browser import BrowserController
from actions.vision import PageAnalyzer, Suggestion
from actions.overlay import Overlay
from actions.config import OPENAI_API_KEY


class ActionEngine:
    """Core loop: screenshot -> vision API -> display suggestions -> execute."""

    def __init__(self, overlay=None):
        self.browser = BrowserController()
        self.analyzer = PageAnalyzer(api_key=OPENAI_API_KEY)
        self.overlay = overlay if overlay is not None else Overlay()
        self._suggestions: list[Suggestion] = []
        self._elements: list[dict] = []  # interactive elements from DOM
        self._selected: int = 0
        self._running: bool = False

    async def start(self, url: str):
        """Launch browser, navigate to URL, and enter the main loop."""
        await self.browser.launch()
        await self.browser.goto(url)
        self._running = True

    async def run_cycle(self) -> list[Suggestion]:
        """Run one screenshot -> analyze -> display cycle. Returns suggestions."""
        print("\n  Analyzing page...", flush=True)

        screenshot = await self.browser.screenshot()
        if screenshot[:8] == b'\x89PNG\r\n\x1a\n':
            w = struct.unpack('>I', screenshot[16:20])[0]
            h = struct.unpack('>I', screenshot[20:24])[0]
            print(f"  Screenshot: {w}x{h}px", flush=True)

        current_url = await self.browser.get_url()
        self._elements = await self.browser.get_interactive_elements()
        print(f"  Found {len(self._elements)} interactive elements", flush=True)

        self._suggestions = await self.analyzer.analyze(
            screenshot, current_url, self._elements
        )
        self._selected = 0
        self.overlay.show(self._suggestions, self._selected)
        return self._suggestions

    def move_selection(self, direction: str):
        """Move the highlight up or down and redisplay."""
        if not self._suggestions:
            return
        if direction == "up":
            self._selected = max(0, self._selected - 1)
        else:
            self._selected = min(len(self._suggestions) - 1, self._selected + 1)
        self.overlay.show(self._suggestions, self._selected)

    def select_index(self, index: int):
        """Jump selection to a specific index."""
        if self._suggestions and 0 <= index < len(self._suggestions):
            self._selected = index
            self.overlay.show(self._suggestions, self._selected)

    async def execute_selected(self):
        """Execute the currently highlighted suggestion."""
        if not self._suggestions:
            return
        suggestion = self._suggestions[self._selected]
        await self.execute(suggestion)

    def _find_element(self, element_id: int) -> dict | None:
        """Look up an interactive element by its ID."""
        for el in self._elements:
            if el["id"] == element_id:
                return el
        return None

    async def execute(self, suggestion: Suggestion):
        """Dispatch a suggestion to the browser."""
        detail = suggestion.action_detail
        print(f"\n  Executing: [{suggestion.action_type}] {suggestion.label}", flush=True)
        print(f"  Detail: {detail}", flush=True)
        try:
            if suggestion.action_type == "click":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    print(f"  Clicking element [{el['id']}] \"{el['text']}\" at ({el['cx']},{el['cy']})", flush=True)
                    await self.browser.click_coords(el["cx"], el["cy"])
                else:
                    print("  Element not found!", flush=True)

            elif suggestion.action_type == "type":
                el = self._find_element(detail.get("element_id", -1))
                if el:
                    print(f"  Clicking input [{el['id']}] at ({el['cx']},{el['cy']})", flush=True)
                    await self.browser.click_coords(el["cx"], el["cy"])
                    await asyncio.sleep(0.3)
                await self.browser.page_type(detail.get("text", ""))

            elif suggestion.action_type == "navigate":
                url = detail.get("url", "")
                print(f"  Navigating to: {url}", flush=True)
                await self.browser.goto(url)

            elif suggestion.action_type == "scroll":
                await self.browser.scroll(detail.get("direction", "down"))

            elif suggestion.action_type == "press_key":
                await self.browser.press_key(detail["key"])

            print("  Done!", flush=True)
        except Exception as e:
            print(f"\n  Action failed: {e}", flush=True)
        # Brief pause to let the page update
        await asyncio.sleep(0.5)

    async def stop(self):
        """Clean up browser resources."""
        self._running = False
        await self.browser.close()
        self.overlay.clear()
