"""Pyppeteer browser controller for taking screenshots and executing actions."""

import asyncio
from pyppeteer import launch

from actions.config import BROWSER_HEADLESS, VIEWPORT_WIDTH, VIEWPORT_HEIGHT


class BrowserController:
    """Controls a Chromium browser via Pyppeteer."""

    def __init__(self):
        self._browser = None
        self._page = None

    async def launch(self):
        """Launch Chromium browser."""
        self._browser = await launch(
            headless=BROWSER_HEADLESS,
            args=[
                f"--window-size={VIEWPORT_WIDTH},{VIEWPORT_HEIGHT}",
                "--no-sandbox",
            ],
        )
        self._page = (await self._browser.pages())[0]
        # Clear the fixed viewport override so the page follows window size
        await self._page._client.send("Emulation.clearDeviceMetricsOverride")

    async def _sync_viewport(self):
        """Sync the viewport to match the actual window content area."""
        try:
            # Get the actual window bounds via CDP
            resp = await self._page._client.send("Browser.getWindowForTarget")
            window_id = resp["windowId"]
            bounds_resp = await self._page._client.send(
                "Browser.getWindowBounds", {"windowId": window_id}
            )
            bounds = bounds_resp["bounds"]
            # Get chrome (toolbar) height by comparing outer vs inner
            outer_h = bounds["height"]
            outer_w = bounds["width"]
            inner = await self._page.evaluate(
                "() => ({w: window.innerWidth, h: window.innerHeight})"
            )
            # If inner matches outer roughly, viewport is already synced
            # Otherwise update it
            if abs(inner["w"] - outer_w) > 10 or abs(inner["h"] - (outer_h - (outer_h - inner["h"]))) > 10:
                await self._page.setViewport(
                    {"width": inner["w"], "height": inner["h"]}
                )
        except Exception:
            pass

    async def get_viewport_size(self) -> tuple[int, int]:
        """Get the current actual viewport size from the browser."""
        size = await self._page.evaluate(
            "() => ({w: window.innerWidth, h: window.innerHeight})"
        )
        return int(size["w"]), int(size["h"])

    async def screenshot(self) -> bytes:
        """Take a screenshot at current window size."""
        # Ensure viewport matches window before capturing
        await self._sync_viewport()
        return await self._page.screenshot({"type": "png"})

    async def get_interactive_elements(self) -> list[dict]:
        """Extract all visible interactive elements with their bounding boxes.

        Returns a list of dicts: {id, tag, text, href, type, cx, cy, w, h}
        where cx,cy is the center of the element's bounding box.
        """
        elements = await self._page.evaluate("""() => {
            const results = [];
            const seen = new Set();
            const selectors = 'a, button, input, textarea, select, [role="button"], [onclick], [tabindex]';
            const els = document.querySelectorAll(selectors);
            let id = 0;
            for (const el of els) {
                const rect = el.getBoundingClientRect();
                // Skip invisible/off-screen elements
                if (rect.width < 5 || rect.height < 5) continue;
                if (rect.top > window.innerHeight || rect.bottom < 0) continue;
                if (rect.left > window.innerWidth || rect.right < 0) continue;

                const text = (el.innerText || el.value || el.getAttribute('aria-label') || el.title || el.placeholder || '').trim().substring(0, 80);
                if (!text && el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA') continue;

                // Deduplicate by position
                const key = `${Math.round(rect.left)},${Math.round(rect.top)}`;
                if (seen.has(key)) continue;
                seen.add(key);

                results.push({
                    id: id++,
                    tag: el.tagName.toLowerCase(),
                    text: text,
                    href: el.href || '',
                    type: el.type || '',
                    cx: Math.round(rect.left + rect.width / 2),
                    cy: Math.round(rect.top + rect.height / 2),
                    w: Math.round(rect.width),
                    h: Math.round(rect.height)
                });
                if (id >= 50) break;  // cap at 50 elements
            }
            return results;
        }""")
        return elements

    async def click(self, selector: str):
        """Click an element by CSS selector."""
        await self._page.click(selector)

    async def click_coords(self, x: int, y: int):
        """Click at specific pixel coordinates."""
        await self._page.mouse.click(x, y)

    async def scroll(self, direction: str):
        """Scroll the page up or down."""
        delta = -300 if direction == "up" else 300
        await self._page.evaluate(f"window.scrollBy(0, {delta})")

    async def goto(self, url: str):
        """Navigate to a URL."""
        try:
            await self._page.goto(url, {"waitUntil": "domcontentloaded", "timeout": 15000})
        except Exception:
            pass
        await asyncio.sleep(1)

    async def type_text(self, selector: str, text: str):
        """Type text into an input element."""
        await self._page.type(selector, text)

    async def page_type(self, text: str):
        """Type text into whatever element is currently focused."""
        await self._page.keyboard.type(text)

    async def press_key(self, key: str):
        """Press a keyboard key (e.g. 'Enter', 'Tab')."""
        await self._page.keyboard.press(key)

    async def get_url(self) -> str:
        """Get the current page URL."""
        return self._page.url

    async def close(self):
        """Close the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
