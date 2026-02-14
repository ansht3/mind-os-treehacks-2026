"""Real-time inference engine with debounce and confidence gating.

Loads a trained model (Pipeline with StandardScaler + LDA + Classifier) and
classifies segments in real-time, applying confidence threshold and
per-command cooldown. Uses TD10 features by default.
"""

import time
import numpy as np
from typing import Optional

from emg_core.dsp.features import extract_features
from emg_core.dsp.filters import preprocess_multichannel
from emg_core.ml.model_io import load_model
from emg_core.api.schemas import Prediction
from emg_core import config


class InferenceEngine:
    """Real-time sEMG classifier with debounce.

    Usage:
        engine = InferenceEngine("demo1")
        prediction = engine.predict(segment_array)
        if prediction:
            print(prediction.cmd, prediction.p)
    """

    def __init__(
        self,
        user_id: str,
        confidence_threshold: float = config.CONFIDENCE_THRESHOLD,
        cooldown_ms: int = config.COOLDOWN_MS,
    ):
        model_data = load_model(user_id)
        self._model = model_data["model"]  # Pipeline (scaler + classifier)
        self._labels: list[str] = model_data["labels"]
        self._threshold = confidence_threshold
        self._cooldown_ms = cooldown_ms

        # Per-command last-fired timestamps
        self._last_fired: dict[str, float] = {}

    @property
    def labels(self) -> list[str]:
        return self._labels

    def predict(self, segment: np.ndarray) -> Optional[Prediction]:
        """Classify a segment and apply debounce logic.

        Args:
            segment: 2D array (num_samples, num_channels) -- raw or preprocessed.

        Returns:
            Prediction if confidence is above threshold and cooldown has passed,
            otherwise None.
        """
        now = time.time()

        # Preprocess with bandpass (1.3-50 Hz, matching AlterEgo paper)
        seg = preprocess_multichannel(
            segment.astype(np.float64),
            fs=config.SAMPLE_RATE,
            apply_bandpass=True,
        )

        # Extract features (time-domain + MFCC)
        features = extract_features(seg, sample_rate=config.SAMPLE_RATE).reshape(1, -1)

        # Predict (Pipeline handles scaling internally)
        proba = self._model.predict_proba(features)[0]
        best_idx = int(np.argmax(proba))
        best_prob = float(proba[best_idx])
        best_cmd = self._labels[best_idx]

        # Confidence gate
        if best_prob < self._threshold:
            return None

        # Cooldown gate
        last = self._last_fired.get(best_cmd, 0)
        if (now - last) * 1000 < self._cooldown_ms:
            return None

        # Fire!
        self._last_fired[best_cmd] = now

        return Prediction(
            t=now,
            cmd=best_cmd,
            p=best_prob,
            cooldown_ms=self._cooldown_ms,
        )

    def predict_raw(self, segment: np.ndarray) -> tuple[str, float, list[float]]:
        """Classify without debounce -- returns (cmd, confidence, all_proba).

        Useful for debugging and display.
        """
        seg = preprocess_multichannel(
            segment.astype(np.float64),
            fs=config.SAMPLE_RATE,
            apply_bandpass=True,
        )
        features = extract_features(seg, sample_rate=config.SAMPLE_RATE).reshape(1, -1)
        proba = self._model.predict_proba(features)[0]
        best_idx = int(np.argmax(proba))
        return self._labels[best_idx], float(proba[best_idx]), proba.tolist()
