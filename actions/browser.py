"""Playwright browser controller for automation and actions."""

import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import Stealth

from actions.config import (
    BROWSER_HEADLESS, VIEWPORT_WIDTH, VIEWPORT_HEIGHT,
    CURSOR_ENABLED, BROWSER_USER_DATA_DIR, TYPE_DELAY_MS,
)
from actions.cursor import PageCursor

# Extra JS injected before every page to mask Chromium automation signals
_STEALTH_INIT_SCRIPT = """
// Mask webdriver flag
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});

// Fake chrome runtime (missing in headless / automation Chromium)
if (!window.chrome) {
    window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){} };
}

// Fake plugins (Chromium automation has empty plugin list)
Object.defineProperty(navigator, 'plugins', {
    get: () => {
        const plugins = [
            {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer', description: 'Portable Document Format'},
            {name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: ''},
            {name: 'Native Client', filename: 'internal-nacl-plugin', description: ''},
        ];
        plugins.item = i => plugins[i] || null;
        plugins.namedItem = n => plugins.find(p => p.name === n) || null;
        plugins.refresh = () => {};
        return plugins;
    }
});

// Fake languages
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});

// Hide automation-related properties from window
delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;

// Permissions API — make "notifications" look like a real browser (default "prompt")
const origQuery = window.Permissions?.prototype?.query;
if (origQuery) {
    window.Permissions.prototype.query = function(params) {
        if (params?.name === 'notifications') {
            return Promise.resolve({state: Notification.permission});
        }
        return origQuery.call(this, params);
    };
}

// WebGL vendor/renderer — hide SwiftShader which is a dead giveaway
const getParameterOrig = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(param) {
    if (param === 37445) return 'Google Inc. (Apple)';           // UNMASKED_VENDOR_WEBGL
    if (param === 37446) return 'ANGLE (Apple, ANGLE Metal Renderer: Apple M1, Unspecified Version)'; // UNMASKED_RENDERER_WEBGL
    return getParameterOrig.call(this, param);
};
const getParameterOrig2 = WebGL2RenderingContext?.prototype?.getParameter;
if (getParameterOrig2) {
    WebGL2RenderingContext.prototype.getParameter = function(param) {
        if (param === 37445) return 'Google Inc. (Apple)';
        if (param === 37446) return 'ANGLE (Apple, ANGLE Metal Renderer: Apple M1, Unspecified Version)';
        return getParameterOrig2.call(this, param);
    };
}
"""


class BrowserController:
    """Controls a Chromium browser via Playwright with stealth patches."""

    def __init__(self, on_cursor_move=None):
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._on_cursor_move = on_cursor_move
        self._cursor: PageCursor | None = None

    async def launch(self):
        """Launch Chromium with stealth patches to avoid bot detection."""
        self._playwright = await async_playwright().start()

        self._context = await self._playwright.chromium.launch_persistent_context(
            user_data_dir=BROWSER_USER_DATA_DIR,
            headless=BROWSER_HEADLESS,
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
            no_viewport=False,
            locale="en-US",
            timezone_id="America/Los_Angeles",
            args=[
                f"--window-size={VIEWPORT_WIDTH},{VIEWPORT_HEIGHT}",
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--disable-infobars",
            ],
            ignore_default_args=[
                "--enable-automation",
                "--enable-blink-features=IdleDetection",
            ],
        )

        # Inject stealth script before any page JS runs (applies to all navigations)
        await self._context.add_init_script(_STEALTH_INIT_SCRIPT)

        self._page = self._context.pages[0] if self._context.pages else await self._context.new_page()

        # Also apply playwright_stealth for extra coverage
        stealth = Stealth(
            navigator_platform_override="MacIntel",
            navigator_user_agent_override=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            ),
        )
        await stealth.apply_stealth_async(self._page)

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
                const isInput = el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.getAttribute('contenteditable') === 'true' || el.getAttribute('contenteditable') === 'plaintext-only' || el.getAttribute('role') === 'combobox' || el.getAttribute('role') === 'searchbox';
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
                '[contenteditable="true"]',
                '[contenteditable="plaintext-only"]',
                'summary',
                'label[for]',
            ].join(', ');
            document.querySelectorAll(standardSelectors).forEach(addElement);

            // Also find inputs/textareas inside shadow-like wrappers (e.g. Google search)
            document.querySelectorAll('[role="combobox"], [role="search"], [aria-label*="earch"]').forEach(el => {
                // Try the element itself
                addElement(el);
                // Try children — Google nests textarea inside a div
                el.querySelectorAll('input, textarea, [contenteditable]').forEach(addElement);
            });

            const allVisible = document.querySelectorAll('div, span, li, img, svg, p, h1, h2, h3, td, th');
            for (const el of allVisible) {
                if (id >= 60) break;
                const style = window.getComputedStyle(el);
                if (style.cursor === 'pointer' || style.cursor === 'text') {
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
            await self._cursor._center()

    async def focus_and_type(self, text: str):
        """Find the first visible input/textarea on the page, click it, type, and press Enter.
        If no input found, types into whatever is currently focused."""
        pos = await self._page.evaluate("""() => {
            const selectors = 'input[type="text"], input[type="search"], input[type="url"], input[type="email"], input:not([type]), textarea, [contenteditable="true"], [contenteditable="plaintext-only"], [role="combobox"], [role="searchbox"]';
            for (const el of document.querySelectorAll(selectors)) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') continue;
                if (rect.width < 10 || rect.height < 10) continue;
                if (rect.top < 0 || rect.top > window.innerHeight) continue;
                return {x: Math.round(rect.left + rect.width / 2), y: Math.round(rect.top + rect.height / 2)};
            }
            // Fallback: check if something is already focused and typeable
            const active = document.activeElement;
            if (active && active !== document.body) {
                const rect = active.getBoundingClientRect();
                if (rect.width > 5 && rect.height > 5) {
                    return {x: Math.round(rect.left + rect.width / 2), y: Math.round(rect.top + rect.height / 2), already_focused: true};
                }
            }
            return null;
        }""")
        if pos:
            await self.click_coords(pos["x"], pos["y"])
        if text:
            await self.page_type(text)
            await self.keyboard_press("Enter")
        return pos is not None

    async def is_focused_typeable(self) -> bool:
        """Check if the currently focused element is a text input or editable field."""
        return await self._page.evaluate("""() => {
            const el = document.activeElement;
            if (!el || el === document.body) return false;
            const tag = el.tagName;
            if (tag === 'TEXTAREA') return true;
            if (tag === 'INPUT') {
                const t = (el.type || '').toLowerCase();
                return ['text','search','url','email','password','tel','number',''].includes(t);
            }
            if (el.getAttribute('contenteditable') === 'true' || el.getAttribute('contenteditable') === 'plaintext-only') return true;
            const role = el.getAttribute('role');
            if (role === 'combobox' || role === 'searchbox' || role === 'textbox') return true;
            return false;
        }""")

    async def keyboard_press(self, key: str):
        """Press a key (internal helper)."""
        await self._page.keyboard.press(key)

    async def type_text(self, selector: str, text: str):
        """Type text into an input element."""
        await self._page.type(selector, text, delay=TYPE_DELAY_MS)

    async def page_type(self, text: str):
        """Type text into whatever element is currently focused (clears field first).
        Types character-by-character with a human-like delay for smooth animation."""
        # Select all existing text and replace it
        modifier = "Meta" if "darwin" in __import__("sys").platform else "Control"
        await self._page.keyboard.press(f"{modifier}+a")
        await self._page.keyboard.type(text, delay=TYPE_DELAY_MS)

    async def press_key(self, key: str):
        """Press a keyboard key (e.g. 'Enter', 'Tab')."""
        await self._page.keyboard.press(key)

    async def go_back(self):
        """Go back in browser history."""
        await self._page.go_back(wait_until="domcontentloaded", timeout=8000)

    async def go_forward(self):
        """Go forward in browser history."""
        await self._page.go_forward(wait_until="domcontentloaded", timeout=8000)

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
        if self._context:
            await self._context.close()
            self._context = None
            self._page = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
