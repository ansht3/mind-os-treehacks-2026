"""OpenClaw-based action predictor for binary yes/no UX."""

import asyncio
import json
import subprocess

from actions.config import OPENCLAW_SESSION_ID


class OpenClawPredictor:
    """Predicts the next browser action using OpenClaw CLI."""

    def __init__(self, session_id: str = OPENCLAW_SESSION_ID):
        self._session_id = session_id

    async def predict(
        self,
        screenshot_b64: str,
        url: str,
        page_title: str,
        past_actions: list[str] | None = None,
    ) -> dict:
        """Predict the single most likely next browser action.

        Returns dict with keys: label, action_type, and action-specific fields
        (x/y for click, text for type, direction for scroll, url for navigate).
        """
        past = past_actions or []
        past_str = "\n".join(f"- {a}" for a in past[-5:]) if past else "(none)"

        prompt = f"""You are a browser automation assistant. Given a screenshot of a webpage, predict the SINGLE most likely next action the user wants to take.

Current URL: {url}
Page title: {page_title}
Recent actions taken:
{past_str}

Respond with ONLY a JSON object (no markdown, no explanation) in one of these formats:
{{"label": "Click the search bar", "action_type": "click", "x": 640, "y": 162}}
{{"label": "Type 'basketball'", "action_type": "type", "text": "basketball", "x": 640, "y": 162}}
{{"label": "Scroll down", "action_type": "scroll", "direction": "down"}}
{{"label": "Go to homepage", "action_type": "navigate", "url": "https://example.com"}}
{{"label": "Press Enter to search", "action_type": "press_key", "key": "Enter"}}

Pick the action that is most natural as the next step. Be specific in the label."""

        return await self._call_openclaw(prompt, screenshot_b64)

    async def predict_alternative(
        self,
        screenshot_b64: str,
        url: str,
        page_title: str,
        rejected_actions: list[str],
    ) -> dict:
        """Predict an alternative action, excluding rejected ones.

        Returns same dict format as predict().
        """
        rejected_str = "\n".join(f"- {a}" for a in rejected_actions)

        prompt = f"""You are a browser automation assistant. Given a screenshot of a webpage, predict the next action the user wants to take.

Current URL: {url}
Page title: {page_title}

The user REJECTED these actions (do NOT suggest them again):
{rejected_str}

Respond with ONLY a JSON object (no markdown, no explanation) in one of these formats:
{{"label": "Click the search bar", "action_type": "click", "x": 640, "y": 162}}
{{"label": "Type 'basketball'", "action_type": "type", "text": "basketball", "x": 640, "y": 162}}
{{"label": "Scroll down", "action_type": "scroll", "direction": "down"}}
{{"label": "Go to homepage", "action_type": "navigate", "url": "https://example.com"}}
{{"label": "Press Enter to search", "action_type": "press_key", "key": "Enter"}}

Suggest a DIFFERENT action that the user might want. Be specific in the label."""

        return await self._call_openclaw(prompt, screenshot_b64)

    async def _call_openclaw(self, message: str, screenshot_b64: str) -> dict:
        """Call openclaw CLI and parse the JSON response.

        The message (with embedded screenshot) is passed via a temp file to
        avoid OS argument-list-too-long errors from large base64 payloads.
        """
        import tempfile, os

        # Include screenshot as base64 in the message body
        if screenshot_b64:
            message += f"\n\n[Screenshot (base64 PNG)]\n{screenshot_b64}"

        # Write message to a temp file and use shell pipe to avoid ARG_MAX
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="openclaw_msg_")
        try:
            with os.fdopen(tmp_fd, "w") as f:
                f.write(message)

            # Read message from file via shell: openclaw agent ... -m "$(cat file)"
            cmd = [
                "openclaw", "agent",
                "--local",
                "--agent", "main",
                "--json",
                "--session-id", self._session_id,
                "--message", "@" + tmp_path,
            ]

            # First try the @file convention; if openclaw doesn't support it,
            # fall back to piping via stdin with shell
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # If @file isn't understood, fall back to shell pipe
            if result.returncode != 0 and "@" in (result.stderr or ""):
                shell_cmd = (
                    f'openclaw agent --local --agent main --json'
                    f' --session-id {self._session_id}'
                    f' --message "$(cat {tmp_path})"'
                )
                result = await asyncio.to_thread(
                    subprocess.run,
                    shell_cmd,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    shell=True,
                )

            if result.returncode != 0:
                raise RuntimeError(f"openclaw failed: {result.stderr.strip()}")

            raw = result.stdout.strip()
            return self._parse_response(raw)

        except subprocess.TimeoutExpired:
            return self._fallback("Scroll down", "scroll", direction="down")
        except FileNotFoundError:
            raise RuntimeError(
                "openclaw CLI not found. Install it with: pip install openclaw"
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _parse_response(self, raw: str) -> dict:
        """Parse openclaw JSON output into an action dict."""
        # openclaw --json returns JSON; try to extract the action object
        # It may return a wrapper like {"response": "..."} or direct JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in the output
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    return self._fallback("Scroll down", "scroll", direction="down")
            else:
                return self._fallback("Scroll down", "scroll", direction="down")

        # If openclaw wraps the response, try to extract nested JSON
        if "response" in data and isinstance(data["response"], str):
            try:
                data = json.loads(data["response"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Validate required fields
        if "label" not in data or "action_type" not in data:
            return self._fallback("Scroll down", "scroll", direction="down")

        return data

    @staticmethod
    def _fallback(label: str, action_type: str, **kwargs) -> dict:
        return {"label": label, "action_type": action_type, **kwargs}
