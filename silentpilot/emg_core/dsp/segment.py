"""Utterance segmentation for sEMG signals.

Two modes:
1. PTT (Push-to-Talk): external trigger marks start/end of utterance.
2. Energy threshold: auto-detect based on RMS energy.

Both produce fixed-length segments for feature extraction.
"""

import time
import uuid
import numpy as np
from scipy.signal import resample
from typing import Optional

from emg_core.api.schemas import Segment
from emg_core import config


class PTTSegmenter:
    """Push-to-talk segmenter.

    Call `start()` when PTT is pressed, feed samples with `add_sample()`,
    and call `stop()` when PTT is released. Returns a fixed-length Segment.
    """

    def __init__(
        self,
        num_channels: int = config.NUM_CHANNELS,
        fixed_length: int = config.SEGMENT_FIXED_LENGTH,
    ):
        self.num_channels = num_channels
        self.fixed_length = fixed_length

        self._recording = False
        self._buffer: list[list[float]] = []
        self._start_t: float = 0.0
        self._label: Optional[str] = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self, label: Optional[str] = None) -> None:
        """Begin recording a segment."""
        self._recording = True
        self._buffer = []
        self._start_t = time.time()
        self._label = label

    def add_sample(self, channels: list[float]) -> None:
        """Add a sample to the current segment buffer."""
        if self._recording:
            self._buffer.append(channels)

    def stop(self) -> Optional[Segment]:
        """Stop recording and return the segment (resampled to fixed length).

        Returns None if the buffer is too short (< 10 samples).
        """
        if not self._recording:
            return None

        self._recording = False
        end_t = time.time()

        if len(self._buffer) < 10:
            return None

        # Convert to numpy and resample to fixed length
        raw = np.array(self._buffer, dtype=np.float64)  # (N, channels)
        resampled = resample(raw, self.fixed_length, axis=0)

        return Segment(
            segment_id=f"seg_{uuid.uuid4().hex[:8]}",
            start_t=self._start_t,
            end_t=end_t,
            samples=resampled.tolist(),
            label=self._label,
        )

    def reset(self) -> None:
        """Clear the buffer."""
        self._recording = False
        self._buffer = []
        self._label = None


class EnergySegmenter:
    """Auto-segmenter based on RMS energy threshold.

    Detects utterances when the summed RMS across channels exceeds
    a threshold for a minimum duration.
    """

    def __init__(
        self,
        num_channels: int = config.NUM_CHANNELS,
        sample_rate: int = config.SAMPLE_RATE,
        fixed_length: int = config.SEGMENT_FIXED_LENGTH,
        threshold: float = config.ENERGY_THRESHOLD,
        min_duration_s: float = config.ENERGY_MIN_DURATION_S,
        silence_s: float = config.ENERGY_SILENCE_S,
    ):
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.fixed_length = fixed_length
        self.threshold = threshold
        self.min_samples = int(min_duration_s * sample_rate)
        self.silence_samples = int(silence_s * sample_rate)

        self._buffer: list[list[float]] = []
        self._active = False
        self._silence_count = 0
        self._start_t: float = 0.0

    def add_sample(self, channels: list[float]) -> Optional[Segment]:
        """Feed a sample, returns a Segment if an utterance is detected.

        Returns None if no segment boundary detected yet.
        """
        # Compute RMS across channels for this sample
        rms = np.sqrt(np.mean(np.array(channels) ** 2))

        if not self._active:
            if rms > self.threshold:
                # Start of utterance
                self._active = True
                self._buffer = [channels]
                self._silence_count = 0
                self._start_t = time.time()
        else:
            self._buffer.append(channels)

            if rms < self.threshold:
                self._silence_count += 1
            else:
                self._silence_count = 0

            # Check if silence exceeds limit
            if self._silence_count >= self.silence_samples:
                self._active = False
                return self._finalize_segment()

        return None

    def _finalize_segment(self) -> Optional[Segment]:
        """Convert buffer to a fixed-length segment."""
        if len(self._buffer) < self.min_samples:
            self._buffer = []
            return None

        raw = np.array(self._buffer, dtype=np.float64)
        resampled = resample(raw, self.fixed_length, axis=0)

        seg = Segment(
            segment_id=f"seg_{uuid.uuid4().hex[:8]}",
            start_t=self._start_t,
            end_t=time.time(),
            samples=resampled.tolist(),
            label=None,
        )
        self._buffer = []
        return seg

    def reset(self) -> None:
        """Reset the segmenter state."""
        self._active = False
        self._buffer = []
        self._silence_count = 0
