"""FastAPI server for EMG Core.

Provides:
- WebSocket endpoint for live streaming (raw samples, predictions)
- HTTP endpoints for calibration, training, and inference control
- CORS enabled for frontend connection
"""

import os
import json
import time
import asyncio
import numpy as np
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from emg_core.api.schemas import (
    CalibStartRequest, CalibSaveRequest, TrainRequest,
    InferStartRequest, TrainResponse, WSMessage,
    Prediction, RawSample,
)
from emg_core import config


# --- Global State ---

class AppState:
    """Mutable application state shared across endpoints."""

    def __init__(self):
        self.pipeline = None  # set in lifespan
        self.ws_clients: list[WebSocket] = []
        self.calib_recording = False
        self.calib_label: Optional[str] = None
        self.calib_segments: list[dict] = []  # list of {samples, label}
        self.inference_engine = None
        self.inference_running = False

state = AppState()


# --- Lifespan ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the EMG pipeline on server startup."""
    from emg_core.pipeline import Pipeline

    state.pipeline = Pipeline()
    await state.pipeline.start()

    # Start background task to stream data to WebSocket clients
    task = asyncio.create_task(_ws_broadcast_loop())

    yield

    # Cleanup
    task.cancel()
    if state.pipeline:
        await state.pipeline.stop()


app = FastAPI(
    title="SilentPilot EMG Core",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- WebSocket Broadcasting ---

async def _ws_broadcast_loop():
    """Background task that reads from pipeline and broadcasts to WS clients."""
    downsample_counter = 0
    downsample_ratio = max(1, config.SAMPLE_RATE // 30)  # ~30 Hz to UI

    while True:
        if state.pipeline is None:
            await asyncio.sleep(0.1)
            continue

        try:
            # Get the latest sample from the pipeline
            sample = await state.pipeline.get_sample()
            if sample is None:
                await asyncio.sleep(0.004)
                continue

            # Handle calibration recording
            if state.calib_recording and state.pipeline.segmenter.is_recording:
                state.pipeline.segmenter.add_sample(
                    [float(v) for v in sample.ch]
                )

            # Handle live inference
            if state.inference_running and state.inference_engine:
                prediction = await state.pipeline.get_prediction()
                if prediction:
                    await _broadcast(WSMessage(
                        type="prediction",
                        data=prediction.model_dump(),
                    ))

            # Downsample raw data for UI
            downsample_counter += 1
            if downsample_counter >= downsample_ratio:
                downsample_counter = 0
                await _broadcast(WSMessage(
                    type="raw",
                    data=sample.model_dump(),
                ))

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[WS broadcast] Error: {e}")
            await asyncio.sleep(0.1)


async def _broadcast(msg: WSMessage):
    """Send a message to all connected WebSocket clients."""
    data = msg.model_dump_json()
    disconnected = []
    for ws in state.ws_clients:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        state.ws_clients.remove(ws)


# --- WebSocket Endpoint ---

@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Live streaming WebSocket.

    Sends: raw samples (downsampled), predictions, status updates.
    Receives: PTT events from the client.
    """
    await websocket.accept()
    state.ws_clients.append(websocket)

    try:
        while True:
            # Listen for client messages (PTT events)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                msg = json.loads(data)
                await _handle_ws_message(msg)
            except asyncio.TimeoutError:
                pass
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)


async def _handle_ws_message(msg: dict):
    """Handle incoming WebSocket messages (e.g., PTT events)."""
    msg_type = msg.get("type")

    if msg_type == "ptt_start":
        label = msg.get("label")
        if state.pipeline:
            state.pipeline.segmenter.start(label=label)
            if state.calib_recording:
                pass  # recording handled in broadcast loop

    elif msg_type == "ptt_stop":
        if state.pipeline:
            segment = state.pipeline.segmenter.stop()
            if segment:
                if state.calib_recording:
                    state.calib_segments.append({
                        "samples": segment.samples,
                        "label": segment.label or state.calib_label,
                    })
                    await _broadcast(WSMessage(
                        type="calib_progress",
                        data={
                            "label": segment.label or state.calib_label,
                            "count": len([
                                s for s in state.calib_segments
                                if s["label"] == (segment.label or state.calib_label)
                            ]),
                        },
                    ))
                elif state.inference_running and state.inference_engine:
                    # Run inference on the segment
                    seg_array = np.array(segment.samples, dtype=np.float64)
                    prediction = state.inference_engine.predict(seg_array)
                    if prediction:
                        await _broadcast(WSMessage(
                            type="prediction",
                            data=prediction.model_dump(),
                        ))


# --- HTTP Endpoints ---

@app.post("/calib/start")
async def calib_start(req: CalibStartRequest):
    """Begin calibration recording for a command label."""
    state.calib_recording = True
    state.calib_label = req.label
    return {"status": "recording", "label": req.label}


@app.post("/calib/stop")
async def calib_stop():
    """Stop calibration recording."""
    state.calib_recording = False
    # If segmenter is still recording, stop it
    if state.pipeline and state.pipeline.segmenter.is_recording:
        state.pipeline.segmenter.stop()
    return {"status": "stopped", "segments_collected": len(state.calib_segments)}


@app.post("/calib/save")
async def calib_save(req: CalibSaveRequest):
    """Save collected calibration segments to disk."""
    if not state.calib_segments:
        raise HTTPException(400, "No segments collected")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, f"{req.user_id}_calib.npz")

    # Load existing data if it exists (append mode)
    existing_segments = []
    existing_labels = []
    if os.path.exists(path):
        existing = np.load(path, allow_pickle=True)
        existing_segments = list(existing["segments"])
        existing_labels = list(existing["labels"])

    # Append new segments
    for seg in state.calib_segments:
        existing_segments.append(np.array(seg["samples"]))
        existing_labels.append(seg["label"])

    np.savez(
        path,
        segments=np.array(existing_segments, dtype=object),
        labels=np.array(existing_labels),
    )

    count = len(state.calib_segments)
    state.calib_segments = []

    return {
        "status": "saved",
        "path": path,
        "new_segments": count,
        "total_segments": len(existing_segments),
    }


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest):
    """Train a classifier for a user."""
    from emg_core.ml.train import train_model

    try:
        result = train_model(req.user_id)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")


@app.post("/infer/start")
async def infer_start(req: InferStartRequest):
    """Start live inference for a user."""
    from emg_core.ml.infer import InferenceEngine
    from emg_core.ml.model_io import model_exists

    if not model_exists(req.user_id):
        raise HTTPException(404, f"No model for user '{req.user_id}'. Train first.")

    state.inference_engine = InferenceEngine(req.user_id)
    state.inference_running = True

    return {
        "status": "running",
        "user_id": req.user_id,
        "labels": state.inference_engine.labels,
    }


@app.post("/infer/stop")
async def infer_stop():
    """Stop live inference."""
    state.inference_running = False
    state.inference_engine = None
    return {"status": "stopped"}


@app.get("/status")
async def get_status():
    """Get current system status."""
    return {
        "reader": config.EMG_READER,
        "sample_rate": config.SAMPLE_RATE,
        "num_channels": config.NUM_CHANNELS,
        "calibrating": state.calib_recording,
        "calib_label": state.calib_label,
        "calib_segments": len(state.calib_segments),
        "inference_running": state.inference_running,
        "ws_clients": len(state.ws_clients),
    }
