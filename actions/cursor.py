"""Smooth animated cursor overlay injected into the browser page.

Uses a visible arrow cursor (div + inline SVG) that starts at viewport center
and physically animates to each target using requestAnimationFrame.
"""

import asyncio

from actions.config import CURSOR_MOVE_DURATION_MS, CURSOR_CLICK_DELAY_MS

# Inline SVG arrow — red with white outline, classic pointer shape
_SVG_ARROW = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="38" height="38" viewBox="0 0 24 24">'
    '<path d="M2 2 L2 20 L7.5 14.5 L12 22 L15 20.5 L10.5 13 L18 13 Z" '
    'fill="#dc2626" stroke="white" stroke-width="1.5" stroke-linejoin="round"/>'
    '</svg>'
).replace('"', '\\"')

# CSS for cursor — includes subtle blink (gentle opacity pulse)
_CURSOR_CSS = (
    "#__sp_cursor { position:fixed; z-index:2147483647; pointer-events:none;"
    " width:38px; height:38px; filter:drop-shadow(1px 2px 4px rgba(0,0,0,0.45));"
    " left:50vw; top:50vh; animation: __sp_blink 1.8s ease-in-out infinite; }"
    " @keyframes __sp_blink { 0%,100%{opacity:1;} 50%{opacity:0.55;} }"
    " #__sp_ripple { position:fixed; z-index:2147483646; pointer-events:none;"
    " width:40px; height:40px; border-radius:50%;"
    " border:2.5px solid rgba(220,38,38,0.85);"
    " transform:translate(-50%,-50%) scale(0); opacity:0; }"
)

# JS that creates the cursor element — idempotent (safe to call multiple times)
INJECT_JS = """() => {
    if (document.getElementById('__sp_cursor')) return;
    const s = document.createElement('style');
    s.id = '__sp_cursor_style';
    s.textContent = `""" + _CURSOR_CSS.replace('`', '\\`') + """`;
    (document.head || document.documentElement).appendChild(s);
    const d = document.createElement('div');
    d.id = '__sp_cursor';
    d.innerHTML = '""" + _SVG_ARROW + """';
    document.documentElement.appendChild(d);
    const r = document.createElement('div');
    r.id = '__sp_ripple';
    document.documentElement.appendChild(r);
}"""

# Template for animate JS — values are embedded via f-string (avoids pyppeteer arg issues)
_ANIMATE_TPL = """() => {{
    const c = document.getElementById('__sp_cursor');
    if (!c) return;
    const startX = parseFloat(c.style.left) || (window.innerWidth / 2);
    const startY = parseFloat(c.style.top) || (window.innerHeight / 2);
    const tx = {tx}; const ty = {ty}; const dur = {dur};
    const dx = tx - startX; const dy = ty - startY;
    if (Math.abs(dx) < 1 && Math.abs(dy) < 1) {{
        c.style.left = tx + 'px'; c.style.top = ty + 'px'; return;
    }}
    const startTime = performance.now();
    function step(now) {{
        let t = Math.min((now - startTime) / dur, 1);
        t = 1 - Math.pow(1 - t, 3);
        c.style.left = (startX + dx * t) + 'px';
        c.style.top = (startY + dy * t) + 'px';
        if (t < 1) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
}}"""

# Template for ripple JS
_RIPPLE_TPL = """() => {{
    const r = document.getElementById('__sp_ripple');
    if (!r) return;
    r.style.left = '{x}px'; r.style.top = '{y}px';
    r.style.opacity = '1';
    r.style.transform = 'translate(-50%,-50%) scale(0)';
    void r.offsetWidth;
    r.style.transition = 'transform 0.35s ease-out, opacity 0.35s ease-out';
    r.style.transform = 'translate(-50%,-50%) scale(2)';
    r.style.opacity = '0';
    setTimeout(() => {{ r.style.transition = 'none'; }}, 350);
}}"""


def _build_persistence_js(svg_arrow, cursor_css):
    """Build JS that installs a MutationObserver to auto-re-inject the cursor."""
    return """() => {
    if (window.__sp_cursor_observer) return;
    const SVG = '""" + svg_arrow + """';
    const CSS = `""" + cursor_css.replace('`', '\\`') + """`;
    function ensureCursor() {
        if (document.getElementById('__sp_cursor')) return;
        if (!document.getElementById('__sp_cursor_style')) {
            const s = document.createElement('style');
            s.id = '__sp_cursor_style';
            s.textContent = CSS;
            (document.head || document.documentElement).appendChild(s);
        }
        const d = document.createElement('div');
        d.id = '__sp_cursor';
        d.innerHTML = SVG;
        if (window.__sp_cursor_x !== undefined) {
            d.style.left = window.__sp_cursor_x + 'px';
            d.style.top = window.__sp_cursor_y + 'px';
        }
        document.documentElement.appendChild(d);
        if (!document.getElementById('__sp_ripple')) {
            const r = document.createElement('div');
            r.id = '__sp_ripple';
            document.documentElement.appendChild(r);
        }
    }
    window.__sp_cursor_observer = new MutationObserver(ensureCursor);
    window.__sp_cursor_observer.observe(document.documentElement, {childList: true, subtree: true});
    ensureCursor();
}"""


PERSIST_JS = _build_persistence_js(_SVG_ARROW, _CURSOR_CSS)

# JS to update the stored position (so observer re-injects at the right spot)
_STORE_POS_TPL = """() => {{
    window.__sp_cursor_x = {x};
    window.__sp_cursor_y = {y};
}}"""


class PageCursor:
    """Injects and controls a smooth animated cursor on a browser page."""

    def __init__(self, on_position_change=None):
        self._on_position_change = on_position_change
        self._page = None
        self._x = 0
        self._y = 0

    async def attach(self, page):
        """Attach cursor to a page and place at viewport center."""
        self._page = page
        await self._inject_cursor()
        await self._install_persistence()
        await self._center()

    async def _inject_cursor(self):
        """Inject cursor element into the current page."""
        try:
            await self._page.evaluate(INJECT_JS)
        except Exception:
            pass

    async def _install_persistence(self):
        """Install MutationObserver that auto-re-injects cursor if removed."""
        try:
            await self._page.evaluate(PERSIST_JS)
        except Exception:
            pass

    async def _store_position(self):
        """Store current position in window globals so observer can restore it."""
        try:
            await self._page.evaluate(_STORE_POS_TPL.format(x=self._x, y=self._y))
        except Exception:
            pass

    async def _center(self):
        """Place cursor at viewport center (no animation)."""
        try:
            pos = await self._page.evaluate(
                "() => ({x: Math.round(window.innerWidth/2), y: Math.round(window.innerHeight/2)})"
            )
            x, y = pos["x"], pos["y"]
            await self._page.evaluate(f"""() => {{
                const c = document.getElementById('__sp_cursor');
                if (c) {{ c.style.left = '{x}px'; c.style.top = '{y}px'; }}
            }}""")
            self._x = x
            self._y = y
        except Exception:
            pass

    async def ensure_alive(self):
        """Re-inject cursor and persistence observer if removed (e.g. after navigation)."""
        try:
            exists = await self._page.evaluate(
                "() => !!document.getElementById('__sp_cursor')"
            )
            if not exists:
                await self._inject_cursor()
                # Snap to last known position
                await self._page.evaluate(f"""() => {{
                    const c = document.getElementById('__sp_cursor');
                    if (c) {{ c.style.left = '{self._x}px'; c.style.top = '{self._y}px'; }}
                }}""")
            # Always reinstall persistence (navigation wipes JS state)
            await self._install_persistence()
            await self._store_position()
        except Exception:
            await self._inject_cursor()
            await self._install_persistence()

    async def move_to(self, x, y, duration_ms=None):
        """Smoothly animate cursor to (x, y) using rAF in the browser."""
        if duration_ms is None:
            duration_ms = CURSOR_MOVE_DURATION_MS
        await self.ensure_alive()
        js = _ANIMATE_TPL.format(tx=x, ty=y, dur=duration_ms)
        try:
            await self._page.evaluate(js)
        except Exception:
            await self._inject_cursor()
            try:
                await self._page.evaluate(f"""() => {{
                    const c = document.getElementById('__sp_cursor');
                    if (c) {{ c.style.left = '{x}px'; c.style.top = '{y}px'; }}
                }}""")
            except Exception:
                pass
        # Wait for the rAF animation to finish on Python side (+ small buffer)
        await asyncio.sleep(duration_ms / 1000 + 0.05)
        self._x = x
        self._y = y
        await self._store_position()
        if self._on_position_change:
            self._on_position_change(x, y, "move")

    async def click_effect(self):
        """Play a ripple effect at the current cursor position."""
        js = _RIPPLE_TPL.format(x=self._x, y=self._y)
        try:
            await self._page.evaluate(js)
        except Exception:
            pass
        if self._on_position_change:
            self._on_position_change(self._x, self._y, "click")
        await asyncio.sleep(CURSOR_CLICK_DELAY_MS / 1000)

    async def hide(self):
        """Hide the cursor (e.g. before navigation)."""
        try:
            await self._page.evaluate("""() => {
                const c = document.getElementById('__sp_cursor');
                if (c) c.style.display = 'none';
                const r = document.getElementById('__sp_ripple');
                if (r) r.style.display = 'none';
            }""")
        except Exception:
            pass

    async def show(self):
        """Show the cursor again after hiding."""
        try:
            await self._page.evaluate("""() => {
                const c = document.getElementById('__sp_cursor');
                if (c) c.style.display = '';
                const r = document.getElementById('__sp_ripple');
                if (r) r.style.display = '';
            }""")
        except Exception:
            pass
