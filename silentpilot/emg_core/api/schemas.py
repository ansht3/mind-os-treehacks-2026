"""Pydantic models for all EMG Core events and API payloads."""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


# --- Internal Event Types ---

class RawSample(BaseModel):
    """Single multi-channel ADC reading."""
    t: float = Field(description="Unix timestamp (seconds)")
    seq: int = Field(description="Sequence number")
    ch: list[int] = Field(description="Per-channel ADC values (uint16)")


class Segment(BaseModel):
    """A detected utterance segment ready for feature extraction."""
    segment_id: str
    start_t: float
    end_t: float
    samples: list[list[float]]  # shape: [num_samples, num_channels]
    label: Optional[str] = None  # set during calibration


class Prediction(BaseModel):
    """Classifier output after debounce gating."""
    t: float
    cmd: str
    p: float = Field(ge=0, le=1, description="Confidence probability")
    cooldown_ms: int


class CommandEvent(BaseModel):
    """Routed command ready for the agent or direct execution."""
    cmd: str
    confidence: float
    mode: str = Field(description="'DIRECT' or 'AGENT'")
    context: Optional[dict] = None


# --- API Request/Response Types ---

class CalibStartRequest(BaseModel):
    label: str


class CalibSaveRequest(BaseModel):
    user_id: str


class TrainRequest(BaseModel):
    user_id: str


class TrainResponse(BaseModel):
    accuracy: float
    per_class_accuracy: dict[str, float]
    confusion_matrix: list[list[int]]
    labels: list[str]
    num_samples: int


class InferStartRequest(BaseModel):
    user_id: str


class WSMessage(BaseModel):
    """WebSocket message envelope."""
    type: str  # "raw", "segment", "prediction", "status"
    data: dict
