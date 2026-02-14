"""Terminal-based overlay for displaying action suggestions."""

import os


class Overlay:
    """Displays numbered suggestions in the terminal with a selection marker."""

    def __init__(self):
        self._ready = False

    def set_status(self, ready: bool):
        """Set status indicator: green (ready) or red (processing)."""
        self._ready = ready
        if ready:
            print("  \033[92m●\033[0m Ready — select an action", flush=True)
        else:
            print("  \033[91m●\033[0m Processing...", flush=True)

    def show(self, suggestions: list, selected_index: int, smart: bool = False):
        """Clear terminal and display suggestions with selection highlight."""
        self.clear()
        tag = "AI Suggestions" if smart else "Quick Actions"
        status = "\033[92m● Ready\033[0m" if self._ready else "\033[91m● Processing\033[0m"
        print(f"  {status}")
        print("=" * 60)
        print(f"  SilentPilot — {tag}")
        print("=" * 60)
        for i, s in enumerate(suggestions):
            marker = ">>" if i == selected_index else "  "
            print(f"  {marker} [{i}] {s.label}")
            print(f"       {s.description}")
        print("=" * 60)
        print("  0-9: pick | UP/DOWN: navigate | ENTER: execute | q: quit")
        print("=" * 60, flush=True)

    def clear(self):
        """Clear the terminal screen."""
        os.system("cls" if os.name == "nt" else "clear")
