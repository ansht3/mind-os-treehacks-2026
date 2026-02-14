"""Entry point for the EMG-driven browser automation."""

import argparse
import asyncio
import queue
import sys
import traceback
import tty
import termios

from actions.engine import ActionEngine
from actions.config import OPENAI_API_KEY


def read_keyboard_command() -> str:
    """Read a single keypress and return a command string.

    Returns: 'up', 'down', 'select', 'quit', or 'pick:N' for number keys.
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            elif seq == "[B":
                return "down"
        elif ch in ("\r", "\n"):
            return "select"
        elif ch == "q":
            return "quit"
        elif ch.isdigit():
            return f"pick:{ch}"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ""


async def run_keyboard(url: str):
    """Main loop using keyboard input (arrow keys + enter)."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check silentpilot/.env", flush=True)
        return

    engine = ActionEngine()
    try:
        print("  Launching browser...", flush=True)
        await engine.start(url)
        print("  Browser ready!", flush=True)
        while True:
            try:
                await engine.run_cycle()
            except Exception as e:
                print(f"\n  Error during analysis: {e}", flush=True)
                traceback.print_exc()
                print("  Press any key to retry, or 'q' to quit...", flush=True)
                cmd = await asyncio.get_event_loop().run_in_executor(
                    None, read_keyboard_command
                )
                if cmd == "quit":
                    return
                continue
            # Input loop within a single cycle
            while True:
                cmd = await asyncio.get_event_loop().run_in_executor(
                    None, read_keyboard_command
                )
                if not cmd:
                    continue
                print(f"\r  > Got: {cmd}       ", flush=True)
                if cmd == "up":
                    engine.move_selection("up")
                elif cmd == "down":
                    engine.move_selection("down")
                elif cmd == "select":
                    await engine.execute_selected()
                    break  # New cycle after executing
                elif cmd.startswith("pick:"):
                    idx = int(cmd.split(":")[1])
                    engine.select_index(idx)
                    await engine.execute_selected()
                    break
                elif cmd == "quit":
                    print("\n  Goodbye!", flush=True)
                    return
    except Exception as e:
        print(f"\n  Fatal error: {e}", flush=True)
        traceback.print_exc()
    finally:
        await engine.stop()


async def run_emg(url: str):
    """Main loop using EMG sensor input via InferenceEngine."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check silentpilot/.env", flush=True)
        return

    # Import here to avoid hard dependency when using keyboard mode
    sys.path.insert(0, sys.path[0].replace("/actions", "") + "/silentpilot")
    from emg_core.ml.infer import InferenceEngine

    engine = ActionEngine()
    emg = InferenceEngine(user_id="demo1")

    CMD_MAP = {
        "SCROLL": "up",
        "CLICK": "down",
        "CONFIRM": "select",
    }

    try:
        print("  Launching browser...", flush=True)
        await engine.start(url)
        print("  Browser ready!", flush=True)
        while True:
            try:
                await engine.run_cycle()
            except Exception as e:
                print(f"\n  Error during analysis: {e}", flush=True)
                traceback.print_exc()
                await asyncio.sleep(2)
                continue
            while True:
                await asyncio.sleep(0.05)
                cmd = await asyncio.get_event_loop().run_in_executor(
                    None, read_keyboard_command
                )
                if cmd == "up":
                    engine.move_selection("up")
                elif cmd == "down":
                    engine.move_selection("down")
                elif cmd == "select":
                    await engine.execute_selected()
                    break
                elif cmd.startswith("pick:"):
                    idx = int(cmd.split(":")[1])
                    engine.select_index(idx)
                    await engine.execute_selected()
                    break
                elif cmd == "quit":
                    print("\n  Goodbye!", flush=True)
                    return
    except Exception as e:
        print(f"\n  Fatal error: {e}", flush=True)
        traceback.print_exc()
    finally:
        await engine.stop()


async def run_gui_loop(url: str, command_queue: queue.Queue, overlay):
    """Main loop for GUI mode: engine runs in this async loop, commands come from queue."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check silentpilot/.env", flush=True)
        return

    engine = ActionEngine(overlay=overlay)
    loop = asyncio.get_event_loop()
    try:
        print("  Launching browser...", flush=True)
        await engine.start(url)
        print("  Browser ready!", flush=True)
        while True:
            try:
                await engine.run_cycle()
            except Exception as e:
                print(f"\n  Error during analysis: {e}", flush=True)
                traceback.print_exc()
                cmd = await loop.run_in_executor(None, command_queue.get)
                if cmd == "quit":
                    return
                continue
            while True:
                cmd = await loop.run_in_executor(None, command_queue.get)
                if not cmd:
                    continue
                if cmd == "up":
                    engine.move_selection("up")
                elif cmd == "down":
                    engine.move_selection("down")
                elif cmd == "select":
                    await engine.execute_selected()
                    break
                elif cmd.startswith("pick:"):
                    idx = int(cmd.split(":")[1])
                    engine.select_index(idx)
                    await engine.execute_selected()
                    break
                elif cmd == "quit":
                    print("\n  Goodbye!", flush=True)
                    return
    except Exception as e:
        print(f"\n  Fatal error: {e}", flush=True)
        traceback.print_exc()
    finally:
        await engine.stop()


def run_gui(url: str):
    """Run with GUI overlay: asyncio runs in main thread, pumped from tkinter (pyppeteer requires main thread for signals)."""
    import tkinter as tk
    from actions.gui_overlay import GuiOverlay

    command_queue = queue.Queue()
    root = tk.Tk()
    overlay = GuiOverlay(command_queue, root)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(run_gui_loop(url, command_queue, overlay))

    def pump():
        loop.run_until_complete(asyncio.sleep(0))
        if task.done():
            root.quit()
        else:
            root.after(5, pump)

    root.after(5, pump)
    root.mainloop()
    if not task.done():
        task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass
    loop.close()


def main():
    parser = argparse.ArgumentParser(description="SilentPilot Browser Automation")
    parser.add_argument(
        "--mode",
        choices=["keyboard", "emg", "gui"],
        default="keyboard",
        help="Input mode: keyboard (default), emg, or gui",
    )
    parser.add_argument(
        "--url",
        default="https://www.google.com",
        help="Starting URL (default: https://www.google.com)",
    )
    args = parser.parse_args()

    if args.mode == "emg":
        asyncio.run(run_emg(args.url))
    elif args.mode == "gui":
        run_gui(args.url)
    else:
        asyncio.run(run_keyboard(args.url))


if __name__ == "__main__":
    main()
