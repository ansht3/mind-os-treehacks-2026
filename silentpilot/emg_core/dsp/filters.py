"""DSP filters for sEMG signal preprocessing.

All functions operate on numpy arrays.
- For real-time streaming, use the causal (online) variants.
- For offline/segment processing, use the standard (filtfilt) variants.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, lfilter


def remove_dc(signal: np.ndarray, window_samples: int = 250) -> np.ndarray:
    """Remove DC offset by subtracting a rolling mean.

    Args:
        signal: 1D array of samples for a single channel.
        window_samples: Window size in samples (default 250 = 1s at 250Hz).

    Returns:
        DC-removed signal.
    """
    if len(signal) < window_samples:
        return signal - np.mean(signal)

    # Use cumsum trick for fast rolling mean
    cumsum = np.cumsum(signal)
    cumsum = np.insert(cumsum, 0, 0)
    rolling_mean = np.empty_like(signal)

    for i in range(len(signal)):
        start = max(0, i - window_samples + 1)
        rolling_mean[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)

    return signal - rolling_mean


def bandpass(
    signal: np.ndarray,
    fs: float = 250.0,
    low: float = 20.0,
    high: float = 450.0,
    order: int = 4,
) -> np.ndarray:
    """Butterworth bandpass filter for raw EMG.

    Skip this if using MyoWare envelope output.

    Args:
        signal: 1D array of samples.
        fs: Sampling frequency in Hz.
        low: Low cutoff frequency.
        high: High cutoff frequency.
        order: Filter order.

    Returns:
        Bandpass-filtered signal.
    """
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = min(high / nyq, 0.99)  # clamp to Nyquist

    if low_norm >= high_norm or low_norm <= 0:
        return signal  # can't apply filter with these params

    b, a = butter(order, [low_norm, high_norm], btype='band')
    return filtfilt(b, a, signal).astype(signal.dtype)


def notch_60hz(
    signal: np.ndarray,
    fs: float = 250.0,
    freq: float = 60.0,
    Q: float = 30.0,
) -> np.ndarray:
    """Notch filter to remove power line interference.

    Args:
        signal: 1D array of samples.
        fs: Sampling frequency.
        freq: Notch frequency (60 Hz US, 50 Hz EU).
        Q: Quality factor (higher = narrower notch).

    Returns:
        Notch-filtered signal.
    """
    if freq >= fs / 2:
        return signal  # can't notch above Nyquist

    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, signal).astype(signal.dtype)


def smooth(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing.

    Args:
        signal: 1D array of samples.
        window: Window size in samples.

    Returns:
        Smoothed signal.
    """
    if window <= 1 or len(signal) < window:
        return signal

    kernel = np.ones(window) / window
    # Use 'same' mode to keep output length equal to input
    return np.convolve(signal, kernel, mode='same')


def preprocess_channel(
    signal: np.ndarray,
    fs: float = 250.0,
    apply_bandpass: bool = True,
    apply_notch: bool = True,
) -> np.ndarray:
    """Full preprocessing pipeline for a single channel.

    Args:
        signal: 1D float array of raw ADC values.
        fs: Sample rate.
        apply_bandpass: Whether to apply bandpass (skip for envelope sensors).
        apply_notch: Whether to apply 60Hz notch.

    Returns:
        Preprocessed signal.
    """
    sig = signal.astype(np.float64)
    sig = remove_dc(sig, window_samples=int(fs))

    if apply_bandpass:
        sig = bandpass(sig, fs=fs)

    if apply_notch:
        sig = notch_60hz(sig, fs=fs)

    sig = smooth(sig, window=5)
    return sig


def preprocess_multichannel(
    data: np.ndarray,
    fs: float = 250.0,
    apply_bandpass: bool = True,
    apply_notch: bool = True,
) -> np.ndarray:
    """Preprocess all channels.

    Args:
        data: 2D array of shape (num_samples, num_channels).
        fs: Sample rate.

    Returns:
        Preprocessed data of same shape.
    """
    result = np.empty_like(data, dtype=np.float64)
    for ch in range(data.shape[1]):
        result[:, ch] = preprocess_channel(
            data[:, ch], fs=fs,
            apply_bandpass=apply_bandpass,
            apply_notch=apply_notch,
        )
    return result
