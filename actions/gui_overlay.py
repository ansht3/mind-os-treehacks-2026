"""Tkinter GUI overlay for selecting an action by number — sky-blue floating panel."""

import queue
import tkinter as tk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from actions.vision import Suggestion

# ── Color palette — translucent sky-blue ──────────────────────
BG          = "#e8f4fc"   # very light sky blue
BG_LIGHT    = "#dceef8"   # slightly deeper sky
BG_CARD     = "#d0e8f4"   # card / bubble background
FG          = "#050510"   # near-black text
FG_DIM      = "#1a2840"   # dark navy secondary
ACCENT      = "#0077cc"   # deep sky accent
GREEN       = "#00a843"   # status green
RED         = "#e03030"   # status red
BORDER      = "#b8d8ec"   # subtle border
SELECTED_BG = "#c8e2f8"   # selected row highlight
TITLE_BG    = "#d4eaf6"   # title bar


class GuiOverlay:
    """Displays numbered suggestions; use Up/Down to select, Enter to execute."""

    def __init__(self, command_queue: queue.Queue, root: tk.Tk):
        self._queue = command_queue
        self._root = root
        self._suggestions: list["Suggestion"] = []
        self._selected_index = 0
        self._list_frame: tk.Frame | None = None
        self._labels: list[tk.Label] = []
        self._entry: tk.Entry | None = None
        self._status_label: tk.Label | None = None
        self._status_dot: tk.Label | None = None
        self._is_ready: bool = False
        self._build_ui()

    # ── UI construction ────────────────────────────────────────
    def _build_ui(self):
        root = self._root
        root.title("SilentPilot")
        root.attributes("-topmost", True)
        try:
            root.attributes("-alpha", 0.82)
        except tk.TclError:
            pass
        root.configure(bg=BG)
        root.geometry("360x520")
        root.minsize(300, 400)
        root.resizable(True, True)

        # Position near right edge of screen
        sw = root.winfo_screenwidth()
        root.geometry(f"+{sw - 390}+80")

        # ── Main container ────────────────────────────────────
        main = tk.Frame(root, bg=BG, padx=14, pady=8)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Status row ────────────────────────────────────────
        status_frame = tk.Frame(main, bg=BG)
        status_frame.pack(fill=tk.X, pady=(0, 6))

        self._status_dot = tk.Label(status_frame, text="●", font=("", 10),
                                    fg=RED, bg=BG)
        self._status_dot.pack(side=tk.LEFT, padx=(0, 6))
        self._status_label = tk.Label(status_frame, text="Processing…",
                                      font=("SF Pro Display", 11, "bold"),
                                      fg=FG, bg=BG, anchor=tk.W)
        self._status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Thin separator
        tk.Frame(main, bg=BORDER, height=1).pack(fill=tk.X, pady=(0, 6))

        # ── Suggestion list ───────────────────────────────────
        list_container = tk.Frame(main, bg=BG)
        list_container.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self._list_canvas = tk.Canvas(list_container, bg=BG, highlightthickness=0, bd=0)
        self._list_frame = tk.Frame(self._list_canvas, bg=BG)
        self._list_frame.bind(
            "<Configure>",
            lambda e: self._list_canvas.configure(scrollregion=self._list_canvas.bbox("all")),
        )
        self._canvas_window = self._list_canvas.create_window(
            (0, 0), window=self._list_frame, anchor=tk.NW,
        )
        self._list_canvas.pack(fill=tk.BOTH, expand=True)
        self._list_canvas.bind("<Configure>", self._on_canvas_configure)
        # Mouse-wheel scroll
        self._list_canvas.bind("<MouseWheel>",
                               lambda e: self._list_canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Thin separator
        tk.Frame(main, bg=BORDER, height=1).pack(fill=tk.X, pady=(2, 6))

        # ── Cursor info ───────────────────────────────────────
        self._cursor_label = tk.Label(main, text="Cursor: idle",
                                      font=("Menlo", 9), fg=FG_DIM, bg=BG, anchor=tk.W)
        self._cursor_label.pack(fill=tk.X, pady=(0, 6))

        # ── Goal input row ────────────────────────────────────
        goal_frame = tk.Frame(main, bg=BG_LIGHT, padx=6, pady=6)
        goal_frame.pack(fill=tk.X, pady=(0, 0))

        self._goal_entry = tk.Entry(
            goal_frame, font=("SF Pro Display", 11), bg="#ffffff", fg=FG,
            insertbackground=FG, bd=0, highlightthickness=1,
            highlightcolor=ACCENT, highlightbackground=BORDER,
        )
        self._goal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=4)
        self._goal_entry.bind("<Return>", lambda e: self._on_goal_submit())

        self._goal_btn = tk.Button(
            goal_frame, text="Go", font=("SF Pro Display", 10, "bold"),
            bg=ACCENT, fg="#ffffff", activebackground="#0277bd", activeforeground="#ffffff",
            bd=0, padx=12, pady=2, cursor="hand2",
            command=self._on_goal_submit,
        )
        self._goal_btn.pack(side=tk.LEFT)

        # ── Stop agent button (hidden by default) ────────────
        self._stop_btn = tk.Button(
            main, text="STOP AGENT", font=("SF Pro Display", 12, "bold"),
            bg="#1a1a1a", fg="#ff6666", activebackground="#333333", activeforeground="#ff6666",
            bd=0, pady=8, cursor="hand2",
            command=lambda: self._queue.put("stop_agent"),
        )
        # Not packed yet — shown only during agent runs

        # ── Hidden numeric entry (still accepts pick commands) ─
        self._entry = tk.Entry(root, bg=BG, fg=BG, bd=0, width=1,
                               insertbackground=BG, highlightthickness=0)
        # Don't pack — invisible, just used programmatically

        root.protocol("WM_DELETE_WINDOW", self._on_quit)

        # Global key bindings
        root.bind("<Up>", lambda e: self._queue.put("up"))
        root.bind("<Down>", lambda e: self._queue.put("down"))
        root.bind("<Return>", lambda e: self._queue.put("select"))
        root.bind("<KP_Enter>", lambda e: self._queue.put("select"))

    # ── Canvas resize ─────────────────────────────────────────
    def _on_canvas_configure(self, event):
        self._list_canvas.itemconfig(self._canvas_window, width=event.width)

    # ── Internal callbacks ────────────────────────────────────
    def _on_execute(self):
        if not self._entry:
            return
        s = self._entry.get().strip()
        if s.isdigit():
            self._queue.put(f"pick:{s}")
        else:
            self._queue.put("select")

    def _on_goal_submit(self):
        goal = self._goal_entry.get().strip()
        if goal:
            self._queue.put(f"goal:{goal}")
            self._goal_entry.delete(0, tk.END)

    def _on_quit(self):
        self._queue.put("quit")
        self._root.quit()

    # ── Public API (unchanged signatures) ─────────────────────
    def set_status(self, ready: bool):
        """Set the status indicator: green (ready) or red (processing). Thread-safe."""
        self._root.after(0, self._update_status, ready)

    def _update_status(self, ready: bool):
        self._is_ready = ready
        if self._status_dot:
            self._status_dot.config(fg=GREEN if ready else RED)
        if self._status_label:
            text = "Ready — select an action" if ready else "Processing…"
            self._status_label.config(text=text)

    def show(self, suggestions: list["Suggestion"], selected_index: int, smart: bool = False):
        """Update the GUI with current suggestions and selection (thread-safe)."""
        self._root.after(0, self._update_ui, suggestions, selected_index, smart)

    def _update_ui(self, suggestions: list["Suggestion"], selected_index: int, smart: bool = False):
        self._suggestions = suggestions
        self._selected_index = selected_index
        frame = self._list_frame
        if not frame:
            return
        for w in frame.winfo_children():
            w.destroy()
        self._labels.clear()

        tag = "AI Suggestions" if smart else "Quick Actions"
        tk.Label(frame, text=tag, font=("SF Pro Display", 9, "italic"),
                 fg=FG_DIM, bg=BG, anchor=tk.W).pack(anchor=tk.W, pady=(0, 4))

        for i, s in enumerate(suggestions):
            is_sel = i == selected_index
            row_bg = SELECTED_BG if is_sel else BG
            row = tk.Frame(frame, bg=row_bg, padx=8, pady=4)
            row.pack(fill=tk.X, pady=1)

            idx_fg = ACCENT if is_sel else FG_DIM
            tk.Label(row, text=f"{i}", font=("Menlo", 10, "bold"),
                     fg=idx_fg, bg=row_bg, width=2, anchor=tk.E).pack(side=tk.LEFT, padx=(0, 8))

            text_frame = tk.Frame(row, bg=row_bg)
            text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            lbl = tk.Label(text_frame, text=s.label, font=("Menlo", 11),
                           fg=FG, bg=row_bg, anchor=tk.W)
            lbl.pack(anchor=tk.W)
            self._labels.append(lbl)

            if s.description:
                tk.Label(text_frame, text=s.description, font=("SF Pro Display", 9),
                         fg=FG_DIM, bg=row_bg, anchor=tk.W).pack(anchor=tk.W)

            if is_sel:
                # Selection indicator bar
                tk.Frame(row, bg=ACCENT, width=3).pack(side=tk.RIGHT, fill=tk.Y)

        if self._entry:
            self._entry.delete(0, tk.END)
            if suggestions:
                self._entry.insert(0, str(selected_index))

    def prompt_type_text(self):
        """Show a text input prompt for typing into a field (thread-safe)."""
        self._root.after(0, self._show_type_prompt)

    def _show_type_prompt(self):
        frame = self._list_frame
        if not frame:
            return
        for w in frame.winfo_children():
            w.destroy()
        self._labels.clear()

        tk.Label(frame, text="What do you want to type?",
                 font=("SF Pro Display", 12, "bold"), fg=FG, bg=BG).pack(anchor=tk.W, pady=(10, 8))

        type_entry = tk.Entry(frame, font=("SF Pro Display", 12), bg="#ffffff", fg=FG,
                              insertbackground=FG, bd=0, highlightthickness=1,
                              highlightcolor=ACCENT, highlightbackground=BORDER)
        type_entry.pack(fill=tk.X, pady=(0, 10), padx=(0, 8), ipady=6)

        def _submit():
            text = type_entry.get().strip()
            if text:
                self._queue.put(f"typed:{text}")

        tk.Button(frame, text="Type & Enter", font=("SF Pro Display", 10, "bold"),
                  bg=ACCENT, fg="#ffffff", activebackground="#0277bd", activeforeground="#ffffff",
                  bd=0, padx=14, pady=4, cursor="hand2",
                  command=_submit).pack(anchor=tk.W)
        type_entry.bind("<Return>", lambda e: _submit())
        type_entry.focus_set()

    def update_agent_status(self, text: str):
        """Show agent step info in the status label (thread-safe)."""
        self._root.after(0, self._status_label.config, {"text": text})

    def show_agent_question(self, question: str):
        """Show a question from the agent and wait for user answer (thread-safe)."""
        self._root.after(0, self._show_question_ui, question)

    def _show_question_ui(self, question: str):
        frame = self._list_frame
        if not frame:
            return
        for w in frame.winfo_children():
            w.destroy()
        self._labels.clear()

        # Chat-bubble card
        card = tk.Frame(frame, bg=BG_CARD, padx=14, pady=12)
        card.pack(fill=tk.X, pady=(8, 10), padx=4)

        tk.Label(card, text="Agent needs your input",
                 font=("SF Pro Display", 9, "bold"), fg=ACCENT, bg=BG_CARD,
                 anchor=tk.W).pack(anchor=tk.W, pady=(0, 6))
        tk.Label(card, text=question, wraplength=300,
                 font=("SF Pro Display", 11), fg=FG, bg=BG_CARD,
                 anchor=tk.W, justify=tk.LEFT).pack(anchor=tk.W)

        # Pill-shaped Yes / No buttons
        btn_row = tk.Frame(frame, bg=BG)
        btn_row.pack(fill=tk.X, pady=(0, 10), padx=4)

        pill_cfg = dict(font=("SF Pro Display", 12, "bold"), bd=0,
                        padx=24, pady=8, cursor="hand2")
        yes_btn = tk.Button(btn_row, text="Yes", bg="#1a1a1a", fg="#44ee88",
                            activebackground="#333333", activeforeground="#44ee88",
                            command=lambda: self._queue.put("answer:Yes, go ahead."),
                            **pill_cfg)
        yes_btn.pack(side=tk.LEFT, padx=(0, 10))

        no_btn = tk.Button(btn_row, text="No", bg="#1a1a1a", fg="#ff6666",
                           activebackground="#333333", activeforeground="#ff6666",
                           command=lambda: self._queue.put("answer:No, skip this."),
                           **pill_cfg)
        no_btn.pack(side=tk.LEFT)

        # Text answer row
        ans_frame = tk.Frame(frame, bg=BG, pady=4)
        ans_frame.pack(fill=tk.X, padx=4)

        tk.Label(ans_frame, text="Or type:", font=("SF Pro Display", 9),
                 fg=FG_DIM, bg=BG).pack(side=tk.LEFT, padx=(0, 6))
        ans_entry = tk.Entry(ans_frame, font=("SF Pro Display", 11), bg="#ffffff", fg=FG,
                             insertbackground=FG, bd=0, highlightthickness=1,
                             highlightcolor=ACCENT, highlightbackground=BORDER)
        ans_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6), ipady=4)

        def _submit_answer():
            text = ans_entry.get().strip()
            if text:
                self._queue.put(f"answer:{text}")

        tk.Button(ans_frame, text="Send", font=("SF Pro Display", 10, "bold"),
                  bg="#005599", fg="#ffffff", activebackground="#004488", activeforeground="#ffffff",
                  bd=0, padx=12, pady=4, cursor="hand2",
                  command=_submit_answer).pack(side=tk.LEFT)
        ans_entry.bind("<Return>", lambda e: _submit_answer())
        ans_entry.focus_set()

    def update_cursor_info(self, x, y, action):
        """Update the cursor position display (thread-safe)."""
        self._root.after(0, self._cursor_label.config, {"text": f"Cursor: ({x}, {y}) — {action}"})

    def show_stop_button(self):
        """Show the stop agent button (thread-safe)."""
        self._root.after(0, lambda: self._stop_btn.pack(fill=tk.X, pady=(6, 0)))

    def hide_stop_button(self):
        """Hide the stop agent button (thread-safe)."""
        self._root.after(0, self._stop_btn.pack_forget)

    def clear(self):
        """Clear the list (thread-safe)."""
        self._root.after(0, self._clear_ui)

    def _clear_ui(self):
        self._suggestions = []
        self._labels.clear()
        frame = self._list_frame
        if frame:
            for w in frame.winfo_children():
                w.destroy()
        if self._entry:
            self._entry.delete(0, tk.END)
