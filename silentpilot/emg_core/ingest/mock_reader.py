"""Mock EMG reader that generates realistic synthetic sEMG signals.

Inspired by the AlterEgo paper:
- 7 target areas (mental, inner/outer laryngeal, hyoid, inner/outer infra-orbital, buccal)
  mapped to N channels
- Neuromuscular signals in 1.3-50 Hz band (not traditional 20-450 Hz EMG)
- Per-command signatures have distinct temporal frequency patterns and per-channel
  amplitude/phase profiles, not just static amplitude differences.
- Signals are deterministic per command (+ noise), so the classifier sees consistent patterns.
"""

import asyncio
import time
import numpy as np
from typing import Optional

from emg_core.ingest.base_reader import BaseReader
from emg_core.api.schemas import RawSample
from emg_core import config


# Per-command signal profiles.
# Each command defines per-channel: (amplitude, frequency_hz, phase_offset_rad)
# This creates distinguishable temporal-spectral signatures across channels,
# mimicking how different subvocalizations activate different muscle groups
# at different frequencies (as the paper shows via activation maximization).
# Profiles are designed so each command has a *unique dominant frequency* on its
# strongest channel, making them maximally separable in the MFCC feature space.
# The frequencies span the 1.3-50 Hz neuromuscular band used in the paper.
COMMAND_PROFILES: dict[str, list[tuple[float, float, float]]] = {
    "OPEN":    [(0.95, 3.0, 0.0),  (0.15, 30.0, 1.0), (0.20, 8.0, 0.5),  (0.10, 40.0, 2.0)],
    "SEARCH":  [(0.10, 35.0, 0.3), (0.95, 7.0, 0.0),  (0.20, 25.0, 1.5), (0.15, 12.0, 0.8)],
    "CLICK":   [(0.15, 15.0, 1.2), (0.10, 40.0, 0.0), (0.95, 12.0, 0.0), (0.20, 5.0, 1.9)],
    "SCROLL":  [(0.20, 8.0, 0.5),  (0.15, 18.0, 1.8), (0.10, 35.0, 0.0), (0.95, 20.0, 0.0)],
    "TYPE":    [(0.15, 25.0, 0.0), (0.90, 15.0, 0.0), (0.90, 4.0, 0.9),  (0.10, 30.0, 1.4)],
    "ENTER":   [(0.90, 10.0, 0.0), (0.15, 6.0, 0.0),  (0.10, 45.0, 0.4), (0.90, 8.0, 0.0)],
    "CONFIRM": [(0.10, 42.0, 0.0), (0.90, 22.0, 0.0), (0.15, 8.0, 2.0),  (0.90, 35.0, 0.0)],
    "CANCEL":  [(0.90, 18.0, 0.0), (0.10, 3.0, 0.0),  (0.90, 28.0, 0.0), (0.15, 10.0, 0.4)],
}

# Second harmonic components for richer signals (subvocalizations have harmonic content)
HARMONIC_RATIO = 0.35  # second harmonic is 35% of fundamental

# Baseline noise amplitude (ADC units around center 2048)
NOISE_AMPLITUDE = 15.0
# Activation signal amplitude
SIGNAL_AMPLITUDE = 250.0


class MockReader(BaseReader):
    """Generates synthetic sEMG signals for development without hardware.

    Produces neuromuscular signals in the 1-50 Hz range with per-command
    frequency/amplitude/phase signatures across channels, matching the
    AlterEgo paper's signal characteristics.

    Usage:
        reader = MockReader()
        await reader.connect()
        sample = await reader.read()  # returns RawSample

    PTT control:
        reader.start_utterance("OPEN")   # begin simulated activation
        reader.stop_utterance()           # end activation
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        num_channels: int = config.NUM_CHANNELS,
    ):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self._seq = 0
        self._running = False
        self._interval = 1.0 / sample_rate

        # Utterance state
        self._active_command: Optional[str] = None
        self._activation_start: float = 0.0
        self._ramp_up_s = 0.04    # 40ms ramp up
        self._ramp_down_s = 0.04  # 40ms ramp down

        # Random state
        self._rng = np.random.default_rng(seed=42)

    async def connect(self) -> None:
        self._running = True
        self._seq = 0
        self._rng = np.random.default_rng(seed=None)

    async def disconnect(self) -> None:
        self._running = False

    def start_utterance(self, command: str) -> None:
        """Begin simulating an EMG activation burst for the given command."""
        self._active_command = command
        self._activation_start = time.time()

    def stop_utterance(self) -> None:
        """End the simulated EMG activation burst."""
        self._active_command = None

    def _get_envelope(self) -> float:
        """Trapezoidal activation envelope (0 to 1)."""
        if self._active_command is None:
            return 0.0
        elapsed = time.time() - self._activation_start
        if elapsed < self._ramp_up_s:
            return elapsed / self._ramp_up_s
        return 1.0

    async def read(self) -> RawSample:
        if not self._running:
            raise RuntimeError("MockReader not connected. Call connect() first.")

        await asyncio.sleep(self._interval)

        now = time.time()
        envelope = self._get_envelope()

        channels: list[int] = []
        for ch_idx in range(self.num_channels):
            # Baseline noise (low amplitude, always present)
            noise = self._rng.normal(0, NOISE_AMPLITUDE)

            # Activation signal: per-command frequency/amplitude/phase signature
            signal = 0.0
            if self._active_command and self._active_command in COMMAND_PROFILES:
                profile = COMMAND_PROFILES[self._active_command]
                amp, freq, phase = profile[ch_idx % len(profile)]

                elapsed = now - self._activation_start
                t = elapsed  # time since utterance start

                # Fundamental frequency component
                fundamental = amp * np.sin(2 * np.pi * freq * t + phase)
                # Second harmonic (richer signal, like real neuromuscular signals)
                harmonic = amp * HARMONIC_RATIO * np.sin(2 * np.pi * (2 * freq) * t + phase * 1.5)
                # Third harmonic (subtle)
                harmonic2 = amp * 0.12 * np.sin(2 * np.pi * (3 * freq) * t + phase * 0.7)

                signal = envelope * SIGNAL_AMPLITUDE * (fundamental + harmonic + harmonic2)

                # Add small per-trial noise (10% of signal for slight variation)
                signal += envelope * self._rng.normal(0, SIGNAL_AMPLITUDE * amp * 0.10)

            # Convert to uint16 ADC range centered at 2048
            value = int(2048 + noise + signal)
            value = max(0, min(4095, value))
            channels.append(value)

        sample = RawSample(t=now, seq=self._seq, ch=channels)
        self._seq += 1
        return sample
