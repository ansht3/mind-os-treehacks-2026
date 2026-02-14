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
        # Clear fixed viewport so page follows window size
        await self._page._client.send("Emulation.clearDeviceMetricsOverride")

    async def get_viewport_size(self) -> tuple[int, int]:
        """Get the current actual viewport size from the browser."""
        size = await self._page.evaluate(
            "() => ({w: window.innerWidth, h: window.innerHeight})"
        )
        return int(size["w"]), int(size["h"])

    async def screenshot(self) -> bytes:
        """Take a screenshot at current window size."""
        return await self._page.screenshot({"type": "png"})

    async def get_interactive_elements(self) -> list[dict]:
        """Extract all visible interactive elements with bounding boxes.

        Covers: links, buttons, inputs, textareas, selects, role-based elements,
        elements with click handlers, elements with cursor:pointer, and
        any visible text element that looks clickable.
        """
        elements = await self._page.evaluate("""() => {
            const results = [];
            const seen = new Set();
            let id = 0;
            const vw = window.innerWidth;
            const vh = window.innerHeight;

            function getText(el) {
                // Try multiple sources for element text
                return (
                    el.getAttribute('aria-label') ||
                    el.innerText ||
                    el.value ||
                    el.title ||
                    el.placeholder ||
                    el.alt ||
                    el.getAttribute('data-tooltip') ||
                    ''
                ).trim().substring(0, 80);
            }

            function isVisible(el) {
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                const rect = el.getBoundingClientRect();
                if (rect.width < 3 || rect.height < 3) return false;
                if (rect.bottom < 0 || rect.top > vh || rect.right < 0 || rect.left > vw) return false;
                return true;
            }

            function addElement(el) {
                if (id >= 60) return;
                if (!isVisible(el)) return;
                const rect = el.getBoundingClientRect();
                const cx = Math.round(rect.left + rect.width / 2);
                const cy = Math.round(rect.top + rect.height / 2);

                // Deduplicate by center position (within 5px)
                const key = `${Math.round(cx/5)*5},${Math.round(cy/5)*5}`;
                if (seen.has(key)) return;

                const text = getText(el);
                // Allow inputs/textareas even without text
                const isInput = el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT';
                if (!text && !isInput) return;

                seen.add(key);
                results.push({
                    id: id++,
                    tag: el.tagName.toLowerCase(),
                    text: text || `(${el.type || el.tagName.toLowerCase()} field)`,
                    href: el.href || el.closest('a')?.href || '',
                    type: el.type || '',
                    cx: cx,
                    cy: cy,
                    w: Math.round(rect.width),
                    h: Math.round(rect.height)
                });
            }

            // 1. Standard interactive elements
            const standardSelectors = [
                'a[href]',
                'button',
                'input',
                'textarea',
                'select',
                '[role="button"]',
                '[role="link"]',
                '[role="tab"]',
                '[role="menuitem"]',
                '[role="option"]',
                '[role="checkbox"]',
                '[role="radio"]',
                '[role="switch"]',
                '[role="combobox"]',
                '[role="searchbox"]',
                '[onclick]',
                '[tabindex]:not([tabindex="-1"])',
                'summary',
                'label[for]',
            ].join(', ');
            document.querySelectorAll(standardSelectors).forEach(addElement);

            // 2. Elements with cursor:pointer that weren't caught above
            const allVisible = document.querySelectorAll('div, span, li, img, svg, p, h1, h2, h3, td, th');
            for (const el of allVisible) {
                if (id >= 60) break;
                const style = window.getComputedStyle(el);
                if (style.cursor === 'pointer') {
                    addElement(el);
                }
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
        """Scroll the page up or down with smooth animation."""
        delta = -800 if direction == "up" else 800
        await self._page.evaluate(f"window.scrollBy({{top: {delta}, behavior: 'smooth'}})")

    async def goto(self, url: str):
        """Navigate to a URL."""
        try:
            await self._page.goto(url, {"waitUntil": "domcontentloaded", "timeout": 15000})
        except Exception:
            pass
        await asyncio.sleep(0.5)

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
