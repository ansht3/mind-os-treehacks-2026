/**
 * Smooth animated cursor overlay injected into the browser page.
 *
 * Uses a visible arrow cursor (div + inline SVG) that starts at viewport center
 * and physically animates to each target using requestAnimationFrame.
 * Ported from silentpilot/actions/cursor.py
 */

import { Page } from "playwright";

const CURSOR_MOVE_DURATION_MS = 400;
const CURSOR_CLICK_DELAY_MS = 200;

// Inline SVG arrow — red with white outline, classic pointer shape
const SVG_ARROW = `
<svg xmlns="http://www.w3.org/2000/svg" width="38" height="38" viewBox="0 0 24 24">
<path d="M2 2 L2 20 L7.5 14.5 L12 22 L15 20.5 L10.5 13 L18 13 Z"
fill="#dc2626" stroke="white" stroke-width="1.5" stroke-linejoin="round"/>
</svg>
`.replace(/\n/g, "");

// CSS for cursor — includes subtle blink (gentle opacity pulse)
const CURSOR_CSS = `
#__sp_cursor { position:fixed; z-index:2147483647; pointer-events:none;
 width:38px; height:38px; filter:drop-shadow(1px 2px 4px rgba(0,0,0,0.45));
 left:50vw; top:50vh; animation: __sp_blink 1.8s ease-in-out infinite; }
 @keyframes __sp_blink { 0%,100%{opacity:1;} 50%{opacity:0.55;} }
 #__sp_ripple { position:fixed; z-index:2147483646; pointer-events:none;
 width:40px; height:40px; border-radius:50%;
 border:2.5px solid rgba(220,38,38,0.85);
 transform:translate(-50%,-50%) scale(0); opacity:0; }
`.replace(/\n/g, "");

// JS that creates the cursor element — idempotent (safe to call multiple times)
const INJECT_JS = `(() => {
    if (document.getElementById('__sp_cursor')) return;
    const s = document.createElement('style');
    s.id = '__sp_cursor_style';
    s.textContent = \`${CURSOR_CSS.replace(/`/g, '\\`')}\`;
    (document.head || document.documentElement).appendChild(s);
    const d = document.createElement('div');
    d.id = '__sp_cursor';
    d.innerHTML = '${SVG_ARROW}';
    document.documentElement.appendChild(d);
    const r = document.createElement('div');
    r.id = '__sp_ripple';
    document.documentElement.appendChild(r);
})()`;

// Template for animate JS
const ANIMATE_TPL = (tx: number, ty: number, dur: number) => `(() => {
    const c = document.getElementById('__sp_cursor');
    if (!c) return;
    const startX = parseFloat(c.style.left) || (window.innerWidth / 2);
    const startY = parseFloat(c.style.top) || (window.innerHeight / 2);
    const tx = ${tx}; const ty = ${ty}; const dur = ${dur};
    const dx = tx - startX; const dy = ty - startY;
    if (Math.abs(dx) < 1 && Math.abs(dy) < 1) {
        c.style.left = tx + 'px'; c.style.top = ty + 'px'; return;
    }
    const startTime = performance.now();
    function step(now) {
        let t = Math.min((now - startTime) / dur, 1);
        t = 1 - Math.pow(1 - t, 3);
        c.style.left = (startX + dx * t) + 'px';
        c.style.top = (startY + dy * t) + 'px';
        if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
})()`;

// Template for ripple JS
const RIPPLE_TPL = (x: number, y: number) => `(() => {
    const r = document.getElementById('__sp_ripple');
    if (!r) return;
    r.style.left = '${x}px'; r.style.top = '${y}px';
    r.style.opacity = '1';
    r.style.transform = 'translate(-50%,-50%) scale(0)';
    void r.offsetWidth;
    r.style.transition = 'transform 0.35s ease-out, opacity 0.35s ease-out';
    r.style.transform = 'translate(-50%,-50%) scale(2)';
    r.style.opacity = '0';
    setTimeout(() => { r.style.transition = 'none'; }, 350);
})()`;

export class PageCursor {
    private page: Page | null = null;
    private x: number = 0;
    private y: number = 0;

    async attach(page: Page): Promise<void> {
        this.page = page;
        await this.injectCursor();
        await this.center();
    }

    private async injectCursor(): Promise<void> {
        if (!this.page) return;
        try {
            await this.page.evaluate(INJECT_JS);
        } catch (e) {
            // Ignore injection errors
        }
    }

    private async center(): Promise<void> {
        if (!this.page) return;
        try {
            const pos = await this.page.evaluate(() => ({
                x: Math.round(window.innerWidth / 2),
                y: Math.round(window.innerHeight / 2)
            }));
            this.x = pos.x;
            this.y = pos.y;
            await this.page.evaluate(`(() => {
                const c = document.getElementById('__sp_cursor');
                if (c) { c.style.left = '${this.x}px'; c.style.top = '${this.y}px'; }
            })()`);
        } catch (e) {
            // Ignore errors
        }
    }

    async ensureAlive(): Promise<void> {
        if (!this.page) return;
        try {
            const exists = await this.page.evaluate(() => !!document.getElementById('__sp_cursor'));
            if (!exists) {
                await this.injectCursor();
                // Snap to last known position
                await this.page.evaluate(`(() => {
                    const c = document.getElementById('__sp_cursor');
                    if (c) { c.style.left = '${this.x}px'; c.style.top = '${this.y}px'; }
                })()`);
            }
        } catch (e) {
            await this.injectCursor();
        }
    }

    async moveTo(x: number, y: number, durationMs: number = CURSOR_MOVE_DURATION_MS): Promise<void> {
        if (!this.page) return;
        await this.ensureAlive();
        const js = ANIMATE_TPL(x, y, durationMs);
        try {
            await this.page.evaluate(js);
        } catch (e) {
            await this.injectCursor();
            try {
                // Try simpler move if animation failed
                await this.page.evaluate(`(() => {
                    const c = document.getElementById('__sp_cursor');
                    if (c) { c.style.left = '${x}px'; c.style.top = '${y}px'; }
                })()`);
            } catch (inner) {
                // Ignore
            }
        }

        // Wait for animation (+ buffer)
        await new Promise(resolve => setTimeout(resolve, durationMs + 50));
        this.x = x;
        this.y = y;
    }

    async clickEffect(): Promise<void> {
        if (!this.page) return;
        const js = RIPPLE_TPL(this.x, this.y);
        try {
            await this.page.evaluate(js);
        } catch (e) {
            // Ignore
        }
        await new Promise(resolve => setTimeout(resolve, CURSOR_CLICK_DELAY_MS));
    }

    async hide(): Promise<void> {
        if (!this.page) return;
        try {
            await this.page.evaluate(() => {
                const c = document.getElementById('__sp_cursor');
                if (c) c.style.display = 'none';
                const r = document.getElementById('__sp_ripple');
                if (r) r.style.display = 'none';
            });
        } catch (e) {
            // Ignore
        }
    }

    async show(): Promise<void> {
        if (!this.page) return;
        try {
            await this.page.evaluate(() => {
                const c = document.getElementById('__sp_cursor');
                if (c) c.style.display = '';
                const r = document.getElementById('__sp_ripple');
                if (r) r.style.display = '';
            });
        } catch (e) {
            // Ignore
        }
    }
}
