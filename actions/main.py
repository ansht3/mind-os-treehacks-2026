"""Entry point for MindOS — supports GUI interactive mode and autonomous agent mode."""

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
    """Main loop: EMG thought-class buttons + free entry. No DOM action suggestions."""
    from actions.engine import ActionEngine
    from actions.word_finder import find_words_for_sequence
    from actions.word_disambiguate import pick_best_word, suggest_noun_or_use_dict

    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check silentpilot/.env", flush=True)
        return

    from actions.vision import AgentPlanner

    engine = ActionEngine(overlay=overlay)
    planner = AgentPlanner(api_key=OPENAI_API_KEY)
    loop = asyncio.get_event_loop()

    emg_sequence: list[str] = []
    accumulated_prompt: list[str] = []
    word_candidates: list[str] = []
    word_candidate_index: int = 0

    def _get_previously_added_words() -> list[str]:
        return accumulated_prompt

    try:
        print("  Launching browser...", flush=True)
        try:
            await engine.start(url)
            print("  Browser ready!", flush=True)
        except Exception as e:
            print(f"  Browser launch failed: {e}", flush=True)
            overlay.update_agent_status("No browser — enter a goal below")

        overlay.set_status(True)
        overlay.update_sequence([])

        while True:
            cmd = await loop.run_in_executor(None, command_queue.get)
            if not cmd:
                continue
            if cmd == "quit":
                print("\n  Goodbye!", flush=True)
                return

            if cmd.startswith("emg:"):
                cls = cmd[4:]
                if cls == "REST":
                    overlay.set_status(False)
                    overlay.update_agent_status("Looking up…")

                    page_context = ""
                    try:
                        page_context = f"Title: {await engine.browser.get_page_title()}\nText: {(await engine.browser.get_page_text())[:1200]}"
                    except Exception:
                        pass
                    prev_words = _get_previously_added_words()

                    # Ask OpenAI: is this a noun? Noun → Yes/No only. Not noun → dictionary + Yes/No/Retry
                    noun_word, is_noun = await suggest_noun_or_use_dict(
                        emg_sequence, page_context, prev_words
                    )

                    if is_noun and noun_word:
                        # Proper noun — Yes/No only, no Retry
                        word_candidates = [noun_word]
                        word_candidate_index = 0
                        overlay.show_word_confirmation(noun_word, False)
                        overlay.set_status(True)
                        overlay.update_agent_status("Yes / No")
                    else:
                        # Fall back to dictionary search
                        candidates = find_words_for_sequence(emg_sequence)
                        if not candidates:
                            overlay.show_word_confirmation("(no matches)", False)
                            overlay.update_agent_status("No matches")
                            word_candidates = []
                            emg_sequence = []
                            overlay.update_sequence([])
                            overlay.set_status(True)
                        else:
                            if len(candidates) == 1:
                                suggested = candidates[0]
                            else:
                                suggested = await pick_best_word(
                                    candidates, page_context, prev_words
                                )
                            word_candidates = candidates
                            word_candidate_index = (
                                candidates.index(suggested)
                                if suggested in candidates
                                else 0
                            )
                            overlay.show_word_confirmation(
                                suggested, len(candidates) > 1
                            )
                            overlay.set_status(True)
                            overlay.update_agent_status("Yes / No / Retry")
                else:
                    emg_sequence.append(cls)
                    overlay.update_sequence(emg_sequence)

            elif cmd == "word_yes":
                if word_candidates and 0 <= word_candidate_index < len(word_candidates):
                    word = word_candidates[word_candidate_index]
                    accumulated_prompt.append(word)
                    overlay.update_prompt_display(accumulated_prompt)
                emg_sequence = []
                word_candidates = []
                overlay.clear_content_area()
                overlay.update_sequence([])
                overlay.update_agent_status("Ready")

            elif cmd == "word_no":
                emg_sequence = []
                word_candidates = []
                overlay.clear_content_area()
                overlay.update_sequence([])
                overlay.update_agent_status("Ready")

            elif cmd == "word_retry":
                if len(word_candidates) > 1:
                    word_candidate_index = (word_candidate_index + 1) % len(word_candidates)
                    suggested = word_candidates[word_candidate_index]
                    overlay.show_word_confirmation(suggested, True)

            elif cmd == "goal:run":
                goal = " ".join(accumulated_prompt)
                if goal:
                    await _run_agent_in_gui(engine, planner, overlay, goal, command_queue)
                accumulated_prompt.clear()
                overlay.update_prompt_display([])
                overlay.set_status(True)
                overlay.update_agent_status("Ready")

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

    def _find_el(elements, element_id):
        for el in elements:
            if el["id"] == element_id:
                return el
        return None

    def _find_el_by_text(elements, text_hint):
        if not text_hint:
            return None
        hint_lower = text_hint.lower()
        for el in elements:
            if hint_lower in el.get("text", "").lower():
                return el
        return None

    planner.reset()
    overlay.set_status(False)
    overlay.update_agent_status(f"Agent: {goal}")
    overlay.show_stop_button()
    browser = engine.browser
    max_steps = MAX_AGENT_STEPS
    action_result = ""
    consecutive_failures = 0

    try:
        for step in range(1, max_steps + 1):
            if _check_stop():
                overlay.update_agent_status("Agent stopped by user")
                return

            await browser.ensure_cursor()

            url_before = await browser.get_url()
            page_title = await browser.get_page_title()

            try:
                elements = await browser.get_interactive_elements()
            except Exception:
                await asyncio.sleep(1)
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
                current_url=url_before,
                page_title=page_title,
                elements=elements,
                page_text=page_text,
                step_number=step,
                action_result=action_result,
            )
            action_result = ""

            if _check_stop():
                overlay.update_agent_status("Agent stopped by user")
                return

            if action is None:
                consecutive_failures += 1
                overlay.update_agent_status(f"Agent step {step}: retrying ({consecutive_failures}/3)...")
                if consecutive_failures >= 3:
                    overlay.update_agent_status("Agent: too many failures, stopped")
                    return
                await asyncio.sleep(1)
                continue
            consecutive_failures = 0

            if action.action_type == "done":
                summary = action.action_detail.get("summary", "Done")
                overlay.update_agent_status(f"Agent done: {summary}")
                await asyncio.sleep(1)
                return

            if action.action_type == "confirm":
                planner._messages.append({"role": "user", "content": "Yes, go ahead."})
                continue

            overlay.update_agent_status(f"Agent: {action.description[:50]}")

            if _check_stop():
                overlay.update_agent_status("Agent stopped by user")
                return

            # Execute action with result feedback
            detail = action.action_detail
            try:
                if action.action_type == "click":
                    el = _find_el(elements, detail.get("element_id", -1))
                    if not el:
                        el = _find_el_by_text(elements, action.description)
                    if el:
                        await browser.click_coords(el["cx"], el["cy"])
                        action_result = f"Clicked \"{el['text'][:50]}\""
                    else:
                        action_result = f"ERROR: element_id {detail.get('element_id')} not found"

                elif action.action_type == "type":
                    el = _find_el(elements, detail.get("element_id", -1))
                    if not el:
                        for e in elements:
                            if e["tag"] in ("input", "textarea") or e.get("type") in ("text", "search"):
                                el = e
                                break
                    if el:
                        await browser.click_coords(el["cx"], el["cy"])
                        text = detail.get("text", "")
                        if text:
                            await browser.page_type(text)
                            await browser.press_key("Enter")
                            action_result = f"Typed \"{text}\" and pressed Enter"
                        else:
                            action_result = f"Clicked input \"{el['text'][:30]}\""
                    else:
                        action_result = "ERROR: no input element found"

                elif action.action_type == "navigate":
                    url = detail.get("url", "")
                    if url:
                        await browser.goto(url)
                        action_result = f"Navigated to {url[:60]}"
                    else:
                        action_result = "ERROR: no URL provided"

                elif action.action_type == "scroll":
                    direction = detail.get("direction", "down")
                    await browser.scroll(direction)
                    action_result = f"Scrolled {direction}"

                elif action.action_type == "press_key":
                    key = detail.get("key", "Enter")
                    await browser.press_key(key)
                    action_result = f"Pressed {key}"

                elif action.action_type == "find_text":
                    await browser.scroll("down")
                    action_result = "Scrolled down to find content"

                elif action.action_type == "history":
                    if detail.get("direction") == "back":
                        await browser.go_back()
                        action_result = "Went back"
                    else:
                        await browser.go_forward()
                        action_result = "Went forward"

            except Exception as e:
                action_result = f"ERROR: {e}"
                print(f"  Agent action failed: {e}", flush=True)

            # Wait for page to update
            if action.action_type in ("type", "navigate", "click"):
                await asyncio.sleep(2)
                url_after = await browser.get_url()
                if url_after != url_before:
                    await asyncio.sleep(1)
            else:
                await asyncio.sleep(1)

        overlay.update_agent_status("Agent: max steps reached")
    finally:
        overlay.hide_stop_button()


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
        # Run multiple iterations to drain pending async work
        try:
            loop.run_until_complete(asyncio.sleep(0.01))
        except Exception:
            pass
        if task.done():
            root.quit()
        else:
            root.after(10, pump)

    root.after(10, pump)
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
    parser = argparse.ArgumentParser(description="MindOS Browser Automation")
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
