"""Rolling z-score normalization for sEMG signals.

Handles electrode drift and inter-session variation by maintaining
exponential moving mean and standard deviation per channel.
"""

import numpy as np
from emg_core import config


class RollingNormalizer:
    """Exponential moving z-score normalizer.

    Maintains running statistics per channel and normalizes incoming
    samples to zero mean, unit variance. Uses exponential weighting
    so recent samples matter more than old ones.
    """

    def __init__(
        self,
        num_channels: int = config.NUM_CHANNELS,
        window_s: float = config.NORM_WINDOW_S,
        sample_rate: int = config.SAMPLE_RATE,
    ):
        self.num_channels = num_channels
        # Exponential decay factor: roughly matches a window of `window_s` seconds
        window_samples = window_s * sample_rate
        self.alpha = 2.0 / (window_samples + 1)

        # Running stats
        self._mean = np.zeros(num_channels, dtype=np.float64)
        self._var = np.ones(num_channels, dtype=np.float64)
        self._initialized = False
        self._count = 0

    def update(self, sample: np.ndarray) -> np.ndarray:
        """Update stats and return normalized sample.

        Args:
            sample: 1D array of shape (num_channels,) with raw values.

        Returns:
            Normalized sample (z-scored).
        """
        sample = np.asarray(sample, dtype=np.float64)

        if not self._initialized:
            self._mean = sample.copy()
            self._var = np.ones(self.num_channels, dtype=np.float64)
            self._initialized = True
            self._count = 1
            return np.zeros(self.num_channels)

        self._count += 1

        # Exponential moving average
        delta = sample - self._mean
        self._mean += self.alpha * delta
        self._var = (1 - self.alpha) * (self._var + self.alpha * delta ** 2)

        # Z-score
        std = np.sqrt(np.maximum(self._var, 1e-8))
        normalized = (sample - self._mean) / std

        return normalized

    def normalize_segment(self, segment: np.ndarray) -> np.ndarray:
        """Normalize a full segment using current running stats.

        Args:
            segment: 2D array (num_samples, num_channels).

        Returns:
            Normalized segment of same shape.
        """
        std = np.sqrt(np.maximum(self._var, 1e-8))
        return (segment - self._mean) / std

    def reset(self) -> None:
        """Reset running statistics."""
        self._mean = np.zeros(self.num_channels, dtype=np.float64)
        self._var = np.ones(self.num_channels, dtype=np.float64)
        self._initialized = False
        self._count = 0
