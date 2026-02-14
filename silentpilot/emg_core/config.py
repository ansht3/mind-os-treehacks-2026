"""Central configuration for EMG Core."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Hardware / Ingestion ---
EMG_READER: str = os.getenv("EMG_READER", "mock")  # "mock" or "serial"
SERIAL_PORT: str = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
SERIAL_BAUD: int = int(os.getenv("SERIAL_BAUD", "115200"))
SAMPLE_RATE: int = int(os.getenv("SAMPLE_RATE", "250"))
NUM_CHANNELS: int = int(os.getenv("NUM_CHANNELS", "4"))

# --- DSP ---
# AlterEgo paper uses 1.3-50 Hz 4th-order Butterworth: low-pass to reject
# movement artifacts, high-pass to prevent aliasing. These are neuromuscular
# signals, NOT traditional EMG (which would be 20-450 Hz).
DC_REMOVE_WINDOW_S: float = 1.0  # seconds for DC removal rolling mean
BANDPASS_LOW: float = 1.3
BANDPASS_HIGH: float = 50.0
BANDPASS_ORDER: int = 4
NOTCH_FREQ: float = 60.0  # 60 Hz for US, 50 Hz for EU
NOTCH_Q: float = 30.0
SMOOTH_WINDOW: int = 3  # samples for moving average (smaller to preserve freq content)

# --- TD0/TD10 Feature Extraction (EMG-UKA corpus) ---
TD10_FRAME_SIZE_MS: int = 27   # EMG-UKA frame size in ms
TD10_FRAME_SHIFT_MS: int = 10  # EMG-UKA frame shift in ms
TD10_CONTEXT: int = 10         # ±10 frames context stacking
LDA_COMPONENTS: int = 32       # LDA dimensionality reduction target
CLASSIFIER_TYPE: str = "rf"    # "rf" (RandomForest) or "lr" (LogisticRegression)
RF_N_ESTIMATORS: int = 200     # Number of trees for RandomForest

# --- Normalization ---
NORM_WINDOW_S: float = 3.0  # seconds for rolling z-score window

# --- Segmentation ---
SEGMENT_FIXED_LENGTH: int = 150  # samples (0.6s at 250Hz)
ENERGY_THRESHOLD: float = 0.5  # RMS threshold for energy-based segmentation
ENERGY_MIN_DURATION_S: float = 0.1  # minimum utterance duration
ENERGY_SILENCE_S: float = 0.2  # silence duration to end segment

# --- Classification ---
CONFIDENCE_THRESHOLD: float = 0.75
COOLDOWN_MS: int = 900

# --- Paths ---
DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

# --- Commands ---
COMMANDS: list[str] = [
    "OPEN", "SEARCH", "CLICK", "SCROLL",
    "TYPE", "ENTER", "CONFIRM", "CANCEL",
]

# --- Agent ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
MCP_SERVER_URL: str = os.getenv("MCP_SERVER_URL", "http://localhost:3333/sse")
AGENT_MODEL: str = os.getenv("AGENT_MODEL", "gpt-4o")
