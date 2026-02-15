"""Entry point for SilentPilot — supports GUI interactive mode and autonomous agent mode."""

import argparse
import asyncio
import queue
import traceback

from actions.config import OPENAI_API_KEY, MAX_AGENT_STEPS


# ── Autonomous agent mode ──────────────────────────────────────────

async def run_agent(goal: str, url: str, max_steps: int):
    """Launch browser and run autonomous agent with the given goal."""
    from actions.engine import AutonomousAgent

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check silentpilot/.env", flush=True)
        return

    agent = AutonomousAgent(max_steps=max_steps)
    try:
        print("  Launching browser...", flush=True)
        await agent.start(url)
        print("  Browser ready!\n", flush=True)
        await agent.run(goal)
    except KeyboardInterrupt:
        print("\n  Interrupted by user.", flush=True)
    except Exception as e:
        print(f"\n  Fatal error: {e}", flush=True)
        traceback.print_exc()
    finally:
        await agent.stop()


# ── GUI interactive mode ───────────────────────────────────────────

async def run_gui_loop(url: str, command_queue: queue.Queue, overlay):
    """Main loop for GUI mode: engine runs in this async loop, commands come from queue."""
    from actions.engine import ActionEngine

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check silentpilot/.env", flush=True)
        return

    from actions.vision import AgentPlanner, Suggestion

    engine = ActionEngine(overlay=overlay)
    planner = AgentPlanner(api_key=OPENAI_API_KEY)
    loop = asyncio.get_event_loop()
    try:
        print("  Launching browser...", flush=True)
        try:
            await engine.start(url)
            print("  Browser ready!", flush=True)
        except Exception as e:
            print(f"  Browser launch failed: {e}", flush=True)
            print("  GUI running without browser. Enter a goal or quit.", flush=True)
            overlay.update_agent_status("No browser — enter a goal below")
            # Wait for user commands even without browser
            while True:
                cmd = await loop.run_in_executor(None, command_queue.get)
                if cmd == "quit":
                    return
                if cmd.startswith("goal:"):
                    overlay.update_agent_status(f"Browser not available")
                continue
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
                    s = engine.get_selected()
                    if s and s.action_type in ("type", "type_anywhere") and not s.action_detail.get("text"):
                        overlay.prompt_type_text()
                        continue
                    landed_on_field = await engine.execute_selected()
                    if landed_on_field:
                        # Clicked a text field — auto-prompt for typing
                        overlay.prompt_type_text()
                        continue
                    break
                elif cmd.startswith("typed:"):
                    text = cmd[6:]
                    s = engine.get_selected()
                    if s and s.action_type in ("type", "type_anywhere"):
                        engine.set_type_text(text)
                        await engine.execute_selected()
                    else:
                        # Came from clicking a text field — type directly into focused element
                        await engine.browser.page_type(text)
                        await engine.browser.press_key("Enter")
                    break
                elif cmd.startswith("pick:"):
                    idx = int(cmd.split(":")[1])
                    engine.select_index(idx)
                    s = engine.get_selected()
                    if s and s.action_type in ("type", "type_anywhere") and not s.action_detail.get("text"):
                        overlay.prompt_type_text()
                        continue
                    landed_on_field = await engine.execute_selected()
                    if landed_on_field:
                        overlay.prompt_type_text()
                        continue
                    break
                elif cmd.startswith("goal:"):
                    goal = cmd[5:]
                    await _run_agent_in_gui(engine, planner, overlay, goal, command_queue)
                    break
                elif cmd == "quit":
                    print("\n  Goodbye!", flush=True)
                    return
    except Exception as e:
        print(f"\n  Fatal error: {e}", flush=True)
        traceback.print_exc()
    finally:
        await engine.stop()


async def _run_agent_in_gui(engine, planner, overlay, goal, command_queue):
    """Run the autonomous agent using the existing engine's browser, with GUI updates."""
    from actions.vision import Suggestion

    def _check_stop() -> bool:
        """Drain the queue and return True if user wants to stop."""
        while True:
            try:
                cmd = command_queue.get_nowait()
                if cmd in ("quit", "stop_agent"):
                    return True
            except Exception:
                return False

    planner.reset()
    overlay.set_status(False)
    overlay.update_agent_status(f"Agent: {goal}")
    overlay.show_stop_button()
    browser = engine.browser
    max_steps = MAX_AGENT_STEPS

    try:
        for step in range(1, max_steps + 1):
            if _check_stop():
                overlay.update_agent_status("Agent stopped by user")
                return

            await browser.ensure_cursor()

            current_url = await browser.get_url()
            page_title = await browser.get_page_title()

            try:
                elements = await browser.get_interactive_elements()
            except Exception:
                await asyncio.sleep(0.5)
                try:
                    elements = await browser.get_interactive_elements()
                except Exception:
                    elements = []

            if _check_stop():
                overlay.update_agent_status("Agent stopped by user")
                return

            overlay.update_agent_status(f"Agent step {step}: {page_title[:40]}")

            page_text = await browser.get_page_text()

            action = await planner.decide_next_action(
                goal=goal,
                current_url=current_url,
                page_title=page_title,
                elements=elements,
                page_text=page_text,
                step_number=step,
            )

            if _check_stop():
                overlay.update_agent_status("Agent stopped by user")
                return

            if action is None:
                overlay.update_agent_status(f"Agent step {step}: retrying...")
                await asyncio.sleep(1)
                continue

            if action.action_type == "done":
                summary = action.action_detail.get("summary", "Done")
                overlay.update_agent_status(f"Agent done: {summary}")
                await asyncio.sleep(1)
                return

            if action.action_type == "confirm":
                question = action.action_detail.get("question", "Should I proceed?")
                overlay.update_agent_status(f"Agent asks: {question}")
                overlay.show_agent_question(question)
                loop = asyncio.get_event_loop()
                while True:
                    cmd = await loop.run_in_executor(None, command_queue.get)
                    if cmd in ("quit", "stop_agent"):
                        overlay.update_agent_status("Agent stopped by user")
                        return
                    if cmd.startswith("answer:"):
                        answer = cmd[7:]
                        planner._messages.append({"role": "user", "content": f"User response: {answer}"})
                        overlay.update_agent_status(f"Agent: continuing...")
                        break
                continue

            overlay.update_agent_status(f"Agent: {action.description[:50]}")

            if _check_stop():
                overlay.update_agent_status("Agent stopped by user")
                return

            # Execute action
            detail = action.action_detail
            try:
                if action.action_type == "click":
                    el = _find_el(elements, detail.get("element_id", -1))
                    if el:
                        await browser.click_coords(el["cx"], el["cy"])
                elif action.action_type == "type":
                    el = _find_el(elements, detail.get("element_id", -1))
                    if el:
                        await browser.click_coords(el["cx"], el["cy"])
                    text = detail.get("text", "")
                    if text:
                        await browser.page_type(text)
                elif action.action_type == "navigate":
                    await browser.goto(detail.get("url", ""))
                elif action.action_type == "scroll":
                    await browser.scroll(detail.get("direction", "down"))
                elif action.action_type == "press_key":
                    await browser.press_key(detail.get("key", "Enter"))
                elif action.action_type == "find_text":
                    await browser.find_text(detail.get("text", ""))
                elif action.action_type == "history":
                    if detail.get("direction") == "back":
                        await browser.go_back()
                    else:
                        await browser.go_forward()
            except Exception as e:
                print(f"  Agent action failed: {e}", flush=True)

            await asyncio.sleep(1)

        overlay.update_agent_status("Agent: max steps reached")
    finally:
        overlay.hide_stop_button()


def _find_el(elements, element_id):
    for el in elements:
        if el["id"] == element_id:
            return el
    return None


def run_gui(url: str):
    """Run with GUI overlay: asyncio pumped from tkinter main loop."""
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


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SilentPilot Browser Automation")
    parser.add_argument(
        "--mode",
        choices=["gui", "agent"],
        default="gui",
        help="Mode: gui (interactive, default) or agent (autonomous)",
    )
    parser.add_argument(
        "--url",
        default="https://www.google.com",
        help="Starting URL (default: https://www.google.com)",
    )
    parser.add_argument(
        "goal",
        nargs="?",
        default=None,
        help='Goal prompt for agent mode, e.g. "find the cheapest basketball on amazon"',
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=MAX_AGENT_STEPS,
        help=f"Max agent steps (default: {MAX_AGENT_STEPS})",
    )
    args = parser.parse_args()

    if args.mode == "agent":
        goal = args.goal
        if not goal:
            goal = input("  Enter your goal: ").strip()
            if not goal:
                print("  No goal provided. Exiting.", flush=True)
                return
        asyncio.run(run_agent(goal, args.url, args.max_steps))
    else:
        run_gui(args.url)


if __name__ == "__main__":
    main()
