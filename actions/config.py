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
OPENAI_MODEL: str = "gpt-4o-mini"
BROWSER_HEADLESS: bool = False
NUM_SUGGESTIONS: int = 10
VIEWPORT_WIDTH: int = 1280
VIEWPORT_HEIGHT: int = 800
