"""Computer Use Agent — autonomous browser control via OpenAI's CUA model."""

import asyncio
import base64
from typing import Callable

from openai import OpenAI

from actions.config import CUA_MODEL, CUA_MAX_STEPS


class CUAgent:
    """Runs an autonomous agentic loop using OpenAI's computer-use-preview model."""

    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)

    async def run(
        self,
        task: str,
        browser,
        on_status: Callable[[str], None] | None = None,
    ) -> str:
        """Execute a multi-step browser task autonomously.

        Args:
            task: Natural language description of what to do.
            browser: BrowserController instance.
            on_status: Optional callback called with status text after each step.

        Returns:
            Final text output from the model, or a status message.
        """
        previous_response_id = None

        for step in range(CUA_MAX_STEPS):
            # 1. Screenshot + viewport size
            screenshot_bytes = await browser.screenshot()
            screenshot_b64 = base64.standard_b64encode(screenshot_bytes).decode()
            vw, vh = await browser.get_viewport_size()

            if on_status:
                on_status(f"Step {step + 1}/{CUA_MAX_STEPS}...")

            # 2. Call the CUA model
            if previous_response_id is None:
                # First call: include the task and initial screenshot
                response = await asyncio.to_thread(
                    self._client.responses.create,
                    model=CUA_MODEL,
                    tools=[{
                        "type": "computer_use_preview",
                        "display_width": vw,
                        "display_height": vh,
                        "environment": "browser",
                    }],
                    input=[
                        {"role": "user", "content": task},
                        {
                            "role": "user",
                            "content": [{
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{screenshot_b64}",
                            }],
                        },
                    ],
                    truncation="auto",
                )
            else:
                # Subsequent calls: send new screenshot as computer_call_output
                response = await asyncio.to_thread(
                    self._client.responses.create,
                    model=CUA_MODEL,
                    previous_response_id=previous_response_id,
                    tools=[{
                        "type": "computer_use_preview",
                        "display_width": vw,
                        "display_height": vh,
                        "environment": "browser",
                    }],
                    input=[{
                        "type": "computer_call_output",
                        "call_id": call_id,
                        "output": {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    }],
                    truncation="auto",
                )

            previous_response_id = response.id

            # 3. Parse response items
            computer_call = None
            text_parts = []

            for item in response.output:
                if item.type == "computer_call":
                    computer_call = item
                elif item.type == "text":
                    text_parts.append(item.text)

            # 4. If no computer_call, task is complete
            if computer_call is None:
                result = " ".join(text_parts) if text_parts else "Task completed."
                if on_status:
                    on_status("Done!")
                return result

            # 5. Execute the action and continue
            call_id = computer_call.call_id
            action = computer_call.action
            await self._execute_action(action, browser)

            # Brief pause to let page update
            await asyncio.sleep(0.5)

        if on_status:
            on_status("Reached max steps.")
        return "Reached maximum steps — task may be incomplete."

    async def _execute_action(self, action, browser) -> None:
        """Map a CUA action to browser controller methods."""
        action_type = action.type

        if action_type == "click":
            await browser.click_coords(action.x, action.y)

        elif action_type == "double_click":
            await browser.click_coords(action.x, action.y)
            await asyncio.sleep(0.05)
            await browser.click_coords(action.x, action.y)

        elif action_type == "type":
            await browser.page_type(action.text)

        elif action_type == "scroll":
            scroll_x = getattr(action, "scroll_x", 0)
            scroll_y = getattr(action, "scroll_y", 0)
            page = browser._page
            await page.evaluate(
                f"window.scrollBy({scroll_x}, {scroll_y})"
            )

        elif action_type == "keypress":
            for key in action.keys:
                await browser.press_key(key)

        elif action_type == "drag":
            page = browser._page
            start = action.start_coordinates
            end = action.end_coordinates
            await page.mouse.move(start[0], start[1])
            await page.mouse.down()
            await page.mouse.move(end[0], end[1])
            await page.mouse.up()

        elif action_type == "wait":
            await asyncio.sleep(2)
