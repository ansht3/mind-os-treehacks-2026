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
        self._build_ui()

    def _build_ui(self):
        self._root.title("SilentPilot — Action")
        self._root.minsize(440, 420)
        self._root.resizable(True, True)

        main = ttk.Frame(self._root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text="Suggested actions (↑/↓ select, Enter execute)", font=("", 11, "bold")).pack(anchor=tk.W)

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

    def _on_quit(self):
        self._queue.put("quit")
        self._root.quit()

    def _update_ui(self, suggestions: list["Suggestion"], selected_index: int):
        self._suggestions = suggestions
        self._selected_index = selected_index
        frame = self._list_frame
        if not frame:
            return
        for w in frame.winfo_children():
            w.destroy()
        self._labels.clear()
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

    def show(self, suggestions: list["Suggestion"], selected_index: int):
        """Update the GUI with current suggestions and selection (thread-safe)."""
        self._root.after(0, self._update_ui, suggestions, selected_index)

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
