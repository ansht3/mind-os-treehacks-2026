# SilentPilot — Claude Session Notes

## Project Location
- Repo: `/Users/shawnshen/Downloads/treehacks/` (git root)
- GitHub: `https://github.com/ansht3/treehacks-2026.git`
- Main code: `treehacks/silentpilot/` (Python EMG signal processing + classification)
- EMG dataset: `treehacks/EMG-UKA-Trial-Corpus/` (only on `datasets` branch)

## Git Structure
- **`datasets` branch**: Contains both `EMG-UKA-Trial-Corpus/` AND `silentpilot/` code
- **`algorithm` branch**: Empty (no commits) — do NOT use
- All work should go on `datasets` branch

## What Was Done (Feb 14, 2026 session)

### 1. EMG-UKA TD0/TD10 Feature Pipeline Integration
Replaced the original AlterEgo-inspired features (87 dims) with EMG-UKA TD0/TD10 features.

**Files modified:**

#### `emg_core/dsp/features.py`
- Original `extract_features()` renamed to `extract_features_legacy()` for backward compat
- Added TD0/TD10 pipeline functions:
  - `_double_moving_average(x)` — 9-point double-average → smoothed signal w[n]
  - `_compute_td0_channel()` — 5 features per frame: w_bar, P_w, P_r, z_p, r_bar
  - `_stack_context()` — ±10 frame context stacking (TD0 → TD10)
  - `extract_features_td10()` — frame-level features (420 dims for 4ch)
  - `extract_features_td10_segment()` — segment-level aggregation (mean+std = 840 dims)
- `extract_features()` now calls TD10 pipeline by default

#### `emg_core/config.py`
- Added: `TD10_FRAME_SIZE_MS=27`, `TD10_FRAME_SHIFT_MS=10`, `TD10_CONTEXT=10`
- Added: `LDA_COMPONENTS=32`, `CLASSIFIER_TYPE="rf"`, `RF_N_ESTIMATORS=200`

#### `emg_core/ml/train.py`
- Added `RandomForestClassifier` (default) and `LinearDiscriminantAnalysis`
- Pipeline: `StandardScaler → LDA (n_components=min(32, n_classes-1)) → RF`
- Factored into `_build_pipeline(n_classes)`

#### `emg_core/ml/infer.py`
- Updated docstring only (feature calls auto-use TD10 via updated `extract_features`)

### 2. Security Fix
- `scripts/emguka_llm_decode.py` line 47: Removed hardcoded OpenAI API key
- Replaced with `os.getenv('OPENAI_API_KEY')`

### 3. Test Results
- `scripts/e2e_test.py` (4 commands): **100%** accuracy (target ≥80%)
- `scripts/e2e_test_full.py` (8 commands): **93.3%** accuracy (target ≥90%)
- Both use `MockReader` with synthetic EMG data, not real sensors

## Feature Dimension Math (4 channels, 250Hz)
- TD0 per frame per channel: 5 features
- TD0 per frame all channels: 5 × 4 = 20
- TD10 stacking ±10 frames: 20 × 21 = 420 dimensions
- Segment-level (mean + std): 420 × 2 = 840 dimensions
- After LDA: 32 dimensions (or n_classes - 1 if fewer classes)

## EMG-UKA Corpus Parameters (for reference)
- Corpus path: `~/.cache/kagglehub/datasets/xabierdezuazo/emguka-trial-corpus/versions/1/EMG-UKA-Trial-Corpus`
- Sampling rate: 600 Hz, 6 EMG channels (7 total, first 6 used)
- Bandpass: 1.3–50 Hz, 4th-order Butterworth; Notch: 60 Hz, Q=30
- LLM decode script uses: RF manner classifier (6 classes) + RF phone classifier (41 phones) → GPT-4.1-nano

## Important Notes
- `.env` contains real API keys — never commit it (gitignored)
- `.env.example` exists with placeholder values
- Python environment: `/Users/shawnshen/miniforge3/bin/python` (Python 3.12)
- Dependencies: scikit-learn, scipy, python-dotenv, numpy
- The `algorithm` branch is an orphan with no commits — avoid it
