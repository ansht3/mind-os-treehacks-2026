# SilentPilot

**sEMG-powered silent speech interface for computer control** -- inspired by MIT's AlterEgo.

Place sEMG electrodes along the jaw, think commands, and an AI agent executes them on your computer.

## Architecture

```
EMG Sensors -> MCU -> Python DSP/ML -> FastAPI -> Agent Orchestrator -> MCP Server -> Browser
```

## Quick Start

### 1. EMG Core (Python)

```bash
cd silentpilot
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn emg_core.api.server:app --reload --port 8000
```

### 2. MCP Server

```bash
cd mcp_server
npm install
npx playwright install chromium
npm run dev
```

### 3. Agent

```bash
cd agent
npm install
npm run dev
```

### 4. Frontend

```bash
cd app_ui
npm install
npm run dev
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

- `EMG_READER=mock` for demo mode (no hardware needed)
- `EMG_READER=serial` for real hardware
- `OPENAI_API_KEY` for the agent

## Mock vs Real Hardware

The system uses a plug-and-play `BaseReader` interface. Set `EMG_READER=mock` to use
synthetic signals, or `EMG_READER=serial` to use real sEMG hardware via USB serial.
