"""Tkinter GUI overlay for selecting an action by number."""

import queue
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from actions.vision import Suggestion


class GuiOverlay:
    """Displays numbered suggestions; use Up/Down to select, Enter to execute."""

    def __init__(self, command_queue: queue.Queue, root: tk.Tk):
        self._queue = command_queue
        self._root = root
        self._suggestions: list["Suggestion"] = []
        self._selected_index = 0
        self._list_frame: tk.Frame | None = None
        self._canvas: tk.Canvas | None = None
        self._labels: list[tk.Label] = []
        self._entry: tk.Entry | None = None
        self._status_dot: tk.Canvas | None = None
        self._status_label: tk.Label | None = None
        self._is_ready: bool = False
        self._build_ui()

    def _build_ui(self):
        self._root.title("SilentPilot — Action")
        self._root.minsize(440, 420)
        self._root.resizable(True, True)

        main = ttk.Frame(self._root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # Status bar at top
        status_frame = ttk.Frame(main)
        status_frame.pack(fill=tk.X, pady=(0, 8))
        self._status_dot = tk.Canvas(status_frame, width=16, height=16, highlightthickness=0)
        self._status_dot.pack(side=tk.LEFT, padx=(0, 6))
        self._status_dot.create_oval(2, 2, 14, 14, fill="red", outline="", tags="dot")
        self._status_label = ttk.Label(status_frame, text="Processing...", font=("", 11, "bold"))
        self._status_label.pack(side=tk.LEFT)

        ttk.Label(main, text="Suggested actions (↑/↓ select, Enter execute)", font=("", 10)).pack(anchor=tk.W)

        # Scrollable area for the 10 options
        container = ttk.Frame(main)
        container.pack(fill=tk.BOTH, expand=True, pady=(4, 8))
        self._canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container)
        self._list_frame = ttk.Frame(self._canvas)
        self._list_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas_window = self._canvas.create_window((0, 0), window=self._list_frame, anchor=tk.NW)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.configure(command=self._canvas.yview)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # Resize canvas inner width when window resizes
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        row = ttk.Frame(main)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Action #:").pack(side=tk.LEFT, padx=(0, 6))
        self._entry = ttk.Entry(row, width=4)
        self._entry.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Execute", command=self._on_execute).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Quit", command=self._on_quit).pack(side=tk.LEFT)

        # Goal input for autonomous agent
        goal_frame = ttk.Frame(main)
        goal_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(goal_frame, text="Goal:", font=("", 10)).pack(side=tk.LEFT, padx=(0, 6))
        self._goal_entry = ttk.Entry(goal_frame)
        self._goal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self._goal_btn = ttk.Button(goal_frame, text="Go", command=self._on_goal_submit)
        self._goal_btn.pack(side=tk.LEFT)
        self._goal_entry.bind("<Return>", lambda e: self._on_goal_submit())

        cursor_frame = ttk.Frame(main)
        cursor_frame.pack(fill=tk.X, pady=(8, 0))
        self._cursor_label = ttk.Label(cursor_frame, text="Cursor: idle", font=("Menlo", 9), foreground="gray")
        self._cursor_label.pack(side=tk.LEFT)

        self._root.protocol("WM_DELETE_WINDOW", self._on_quit)

        # Up/Down/Enter work from anywhere in the window
        self._root.bind("<Up>", lambda e: self._queue.put("up"))
        self._root.bind("<Down>", lambda e: self._queue.put("down"))
        self._root.bind("<Return>", lambda e: self._queue.put("select"))
        self._root.bind("<KP_Enter>", lambda e: self._queue.put("select"))
        self._entry.bind("<Return>", lambda e: self._on_execute())

    def _on_canvas_configure(self, event):
        if self._canvas:
            self._canvas.itemconfig(self._canvas_window, width=event.width)

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

    def _update_status(self, ready: bool):
        self._is_ready = ready
        if self._status_dot:
            color = "#22c55e" if ready else "#ef4444"
            self._status_dot.itemconfig("dot", fill=color)
        if self._status_label:
            text = "Ready — select an action" if ready else "Processing..."
            self._status_label.config(text=text)

    def set_status(self, ready: bool):
        """Set the status indicator: green (ready) or red (processing). Thread-safe."""
        self._root.after(0, self._update_status, ready)

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
        header = ttk.Label(frame, text=tag, font=("", 9, "italic"), foreground="gray")
        header.pack(anchor=tk.W, pady=(0, 4))
        for i, s in enumerate(suggestions):
            row = ttk.Frame(frame)
            row.pack(fill=tk.X, pady=2)
            marker = ">>" if i == selected_index else "  "
            lbl = ttk.Label(row, text=f"{marker} [{i}] {s.label}", anchor=tk.W)
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._labels.append(lbl)
            desc = ttk.Label(row, text=s.description, anchor=tk.W, foreground="gray")
            desc.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if self._entry:
            self._entry.delete(0, tk.END)
            if suggestions:
                self._entry.insert(0, str(selected_index))

    def show(self, suggestions: list["Suggestion"], selected_index: int, smart: bool = False):
        """Update the GUI with current suggestions and selection (thread-safe)."""
        self._root.after(0, self._update_ui, suggestions, selected_index, smart)

    def prompt_type_text(self):
        """Show a text input prompt for typing into a field (thread-safe)."""
        self._root.after(0, self._show_type_prompt)

    def _show_type_prompt(self):
        """Replace suggestion list with a text input for typing."""
        frame = self._list_frame
        if not frame:
            return
        for w in frame.winfo_children():
            w.destroy()
        self._labels.clear()

        ttk.Label(frame, text="What do you want to type?", font=("", 11, "bold")).pack(anchor=tk.W, pady=(8, 8))
        type_entry = ttk.Entry(frame, font=("", 12))
        type_entry.pack(fill=tk.X, pady=(0, 8), padx=(0, 8))

        def _submit():
            text = type_entry.get().strip()
            if text:
                self._queue.put(f"typed:{text}")
        ttk.Button(frame, text="Type & Enter", command=_submit).pack(anchor=tk.W)
        type_entry.bind("<Return>", lambda e: _submit())
        type_entry.focus_set()

    def update_agent_status(self, text: str):
        """Show agent step info in the status label (thread-safe)."""
        self._root.after(0, self._status_label.config, {"text": text})

    def show_agent_question(self, question: str):
        """Show a question from the agent and wait for user answer (thread-safe)."""
        self._root.after(0, self._show_question_ui, question)

    def _show_question_ui(self, question: str):
        """Replace suggestion list with a question + answer UI."""
        frame = self._list_frame
        if not frame:
            return
        for w in frame.winfo_children():
            w.destroy()
        self._labels.clear()

        ttk.Label(frame, text="Agent needs your input:", font=("", 10, "bold")).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(frame, text=question, wraplength=380, font=("", 11)).pack(anchor=tk.W, pady=(0, 12))

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(btn_row, text="Yes", command=lambda: self._queue.put("answer:Yes, go ahead.")).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="No", command=lambda: self._queue.put("answer:No, skip this.")).pack(side=tk.LEFT, padx=(0, 8))

        ans_row = ttk.Frame(frame)
        ans_row.pack(fill=tk.X)
        ttk.Label(ans_row, text="Or type:").pack(side=tk.LEFT, padx=(0, 6))
        ans_entry = ttk.Entry(ans_row)
        ans_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        def _submit_answer():
            text = ans_entry.get().strip()
            if text:
                self._queue.put(f"answer:{text}")
        ttk.Button(ans_row, text="Send", command=_submit_answer).pack(side=tk.LEFT)
        ans_entry.bind("<Return>", lambda e: _submit_answer())
        ans_entry.focus_set()

    def update_cursor_info(self, x, y, action):
        """Update the cursor position display (thread-safe)."""
        self._root.after(0, self._cursor_label.config, {"text": f"Cursor: ({x}, {y}) \u2014 {action}"})

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
