"""Tkinter GUI overlay for selecting an action by number."""

import queue
import tkinter as tk
from tkinter import ttk
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from actions.vision import Suggestion


class GuiOverlay:
    """Displays numbered suggestions; use Up/Down to select, Enter to execute."""

    def __init__(self, command_queue: queue.Queue, root: tk.Tk, prediction_mode: bool = False):
        self._queue = command_queue
        self._root = root
        self._prediction_mode = prediction_mode
        self._suggestions: list["Suggestion"] = []
        self._selected_index = 0
        self._list_frame: tk.Frame | None = None
        self._canvas: tk.Canvas | None = None
        self._labels: list[tk.Label] = []
        self._entry: tk.Entry | None = None
        self._task_entry: tk.Entry | None = None
        self._status_dot: tk.Canvas | None = None
        self._status_label: tk.Label | None = None
        self._is_ready: bool = False
        self._prediction_label: tk.Label | None = None
        self._yes_btn: tk.Button | None = None
        self._no_btn: tk.Button | None = None
        self._prediction_frame: tk.Frame | None = None
        self._legacy_frame: tk.Frame | None = None
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

        # --- Prediction mode UI ---
        self._prediction_frame = ttk.Frame(main)
        if self._prediction_mode:
            self._prediction_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        ttk.Label(self._prediction_frame, text="Next Action", font=("", 10)).pack(anchor=tk.W)
        self._prediction_label = ttk.Label(
            self._prediction_frame, text="Waiting for prediction...",
            font=("", 16, "bold"), wraplength=400, justify=tk.LEFT,
        )
        self._prediction_label.pack(anchor=tk.W, pady=(12, 16))

        btn_row = ttk.Frame(self._prediction_frame)
        btn_row.pack(fill=tk.X, pady=(0, 8))
        self._yes_btn = ttk.Button(
            btn_row, text="Yes (1)", command=lambda: self._queue.put("yes"),
        )
        self._yes_btn.pack(side=tk.LEFT, padx=(0, 12), ipadx=20, ipady=8)
        self._no_btn = ttk.Button(
            btn_row, text="No (2)", command=lambda: self._queue.put("no"),
        )
        self._no_btn.pack(side=tk.LEFT, ipadx=20, ipady=8)
        ttk.Button(btn_row, text="Quit", command=self._on_quit).pack(side=tk.RIGHT)

        # --- Legacy mode UI ---
        self._legacy_frame = ttk.Frame(main)
        if not self._prediction_mode:
            self._legacy_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self._legacy_frame, text="Suggested actions (↑/↓ select, Enter execute)", font=("", 10)).pack(anchor=tk.W)

        # Scrollable area for the 10 options
        container = ttk.Frame(self._legacy_frame)
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

        row = ttk.Frame(self._legacy_frame)
        row.pack(fill=tk.X)
        ttk.Label(row, text="Action #:").pack(side=tk.LEFT, padx=(0, 6))
        self._entry = ttk.Entry(row, width=4)
        self._entry.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Execute", command=self._on_execute).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(row, text="Quit", command=self._on_quit).pack(side=tk.LEFT)

        # Task input row for CUA
        task_row = ttk.Frame(self._legacy_frame)
        task_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(task_row, text="AI Task:").pack(side=tk.LEFT, padx=(0, 6))
        self._task_entry = ttk.Entry(task_row)
        self._task_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        ttk.Button(task_row, text="Run Task", command=self._on_run_task).pack(side=tk.LEFT)
        self._task_entry.bind("<Return>", self._on_task_entry_return)

        self._root.protocol("WM_DELETE_WINDOW", self._on_quit)

        # Key bindings depend on mode
        if self._prediction_mode:
            self._root.bind("1", lambda e: self._queue.put("yes"))
            self._root.bind("2", lambda e: self._queue.put("no"))
            self._root.bind("q", lambda e: self._on_quit())
        else:
            self._root.bind("<Up>", lambda e: self._queue.put("up"))
            self._root.bind("<Down>", lambda e: self._queue.put("down"))
            self._root.bind("<Return>", lambda e: self._queue.put("select"))
            self._root.bind("<KP_Enter>", lambda e: self._queue.put("select"))
            self._entry.bind("<Return>", lambda e: (self._on_execute(), "break")[-1])

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

    def _on_task_entry_return(self, event):
        """Handle Enter in task entry — run task and stop event from bubbling to root."""
        self._on_run_task()
        return "break"

    def _on_run_task(self):
        if not self._task_entry:
            return
        text = self._task_entry.get().strip()
        if text:
            self._queue.put(f"cua:{text}")
            self._task_entry.delete(0, tk.END)

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
        # Only update the action entry if the task entry doesn't have focus
        task_has_focus = (self._task_entry and
                          self._root.focus_get() == self._task_entry)
        if self._entry and not task_has_focus:
            self._entry.delete(0, tk.END)
            if suggestions:
                self._entry.insert(0, str(selected_index))

    def show(self, suggestions: list["Suggestion"], selected_index: int, smart: bool = False):
        """Update the GUI with current suggestions and selection (thread-safe)."""
        self._root.after(0, self._update_ui, suggestions, selected_index, smart)

    def _update_prediction(self, label: str):
        if self._prediction_label:
            self._prediction_label.config(text=f"{label}?")

    def show_prediction(self, label: str):
        """Display a single prediction with yes/no controls (thread-safe)."""
        self._root.after(0, self._update_prediction, label)

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
