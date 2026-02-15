"""Playwright browser controller for automation and actions."""

import asyncio
from playwright.async_api import async_playwright

from actions.config import BROWSER_HEADLESS, VIEWPORT_WIDTH, VIEWPORT_HEIGHT, CURSOR_ENABLED
from actions.cursor import PageCursor


class BrowserController:
    """Controls a Chromium browser via Playwright."""

    def __init__(self, on_cursor_move=None):
        self._playwright = None
        self._browser = None
        self._page = None
        self._on_cursor_move = on_cursor_move
        self._cursor: PageCursor | None = None

    async def launch(self):
        """Launch Chromium browser."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=BROWSER_HEADLESS,
            args=[
                f"--window-size={VIEWPORT_WIDTH},{VIEWPORT_HEIGHT}",
                "--no-sandbox",
            ],
        )
        context = await self._browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            no_viewport=False,
        )
        self._page = await context.new_page()

        if CURSOR_ENABLED:
            self._cursor = PageCursor(on_position_change=self._on_cursor_move)
            await self._cursor.attach(self._page)

    async def get_viewport_size(self) -> tuple[int, int]:
        """Get the current actual viewport size from the browser."""
        size = await self._page.evaluate(
            "() => ({w: window.innerWidth, h: window.innerHeight})"
        )
        return int(size["w"]), int(size["h"])

    async def ensure_cursor(self):
        """Re-inject cursor if it was removed by dynamic page updates."""
        if self._cursor:
            await self._cursor.ensure_alive()

    async def screenshot(self) -> bytes:
        """Take a screenshot at current window size."""
        return await self._page.screenshot(type="png")

    async def get_interactive_elements(self) -> list[dict]:
        """Extract all visible interactive elements with bounding boxes."""
        elements = await self._page.evaluate("""() => {
            const results = [];
            const seen = new Set();
            let id = 0;
            const vw = window.innerWidth;
            const vh = window.innerHeight;

            function getText(el) {
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

                const key = `${Math.round(cx/5)*5},${Math.round(cy/5)*5}`;
                if (seen.has(key)) return;

                const text = getText(el);
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
        if self._cursor:
            await self._cursor.move_to(x, y)
            await self._cursor.click_effect()
        await self._page.mouse.click(x, y)
        # Click may trigger navigation — wait briefly then restore cursor
        await asyncio.sleep(0.5)
        await self.ensure_cursor()

    async def scroll(self, direction: str):
        """Scroll the page up or down with smooth animation."""
        if self._cursor:
            w, h = await self.get_viewport_size()
            await self._cursor.move_to(w // 2, h // 2, duration_ms=200)
        delta = -800 if direction == "up" else 800
        await self._page.evaluate(f"window.scrollBy({{top: {delta}, behavior: 'smooth'}})")

    async def goto(self, url: str):
        """Navigate to a URL and wait for page to be interactive."""
        try:
            await self._page.goto(url, wait_until="networkidle", timeout=12000)
        except Exception:
            pass
        await asyncio.sleep(0.5)
        if self._cursor:
            await self._cursor._inject_cursor()
            await self._cursor._install_persistence()
            await self._cursor._center()

    async def type_text(self, selector: str, text: str):
        """Type text into an input element."""
        await self._page.type(selector, text)

    async def page_type(self, text: str):
        """Type text into whatever element is currently focused."""
        await self._page.keyboard.type(text)

    async def press_key(self, key: str):
        """Press a keyboard key (e.g. 'Enter', 'Tab')."""
        await self._page.keyboard.press(key)
        # Enter/submit can trigger navigation — restore cursor
        if key in ("Enter", "Return"):
            await asyncio.sleep(0.5)
            await self.ensure_cursor()

    async def go_back(self):
        """Go back in browser history."""
        await self._page.go_back(wait_until="domcontentloaded", timeout=8000)
        await asyncio.sleep(0.3)
        await self.ensure_cursor()

    async def go_forward(self):
        """Go forward in browser history."""
        await self._page.go_forward(wait_until="domcontentloaded", timeout=8000)
        await asyncio.sleep(0.3)
        await self.ensure_cursor()

    async def get_url(self) -> str:
        """Get the current page URL."""
        return self._page.url

    async def get_page_title(self) -> str:
        """Get the current page title."""
        return await self._page.evaluate("() => document.title") or ""

    async def get_page_text(self) -> str:
        """Get visible text content from the page (truncated)."""
        text = await self._page.evaluate(
            "() => document.body.innerText.substring(0, 3000)"
        )
        return text or ""

    async def close(self):
        """Close the browser."""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
