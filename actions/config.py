"""Configuration for the actions module."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from silentpilot/ dir (where the keys live)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / "silentpilot" / ".env")
# Also try project root .env as fallback
load_dotenv(_project_root / ".env")

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = "gpt-4o"
BROWSER_HEADLESS: bool = False
NUM_SUGGESTIONS: int = 12
VIEWPORT_WIDTH: int = 1280
VIEWPORT_HEIGHT: int = 800

# Persistent browser profile — avoids CAPTCHAs and keeps login sessions
BROWSER_USER_DATA_DIR: str = str(_project_root / ".browser_profile")

# Animated cursor
CURSOR_ENABLED: bool = True
CURSOR_MOVE_DURATION_MS: int = 400
CURSOR_CLICK_DELAY_MS: int = 200

# Typing animation — delay in ms between each keystroke
TYPE_DELAY_MS: int = 45

# Autonomous agent
MAX_AGENT_STEPS: int = 50
