"""Feature extraction for sEMG segments.

Implements two feature pipelines:
1. **Legacy (AlterEgo-inspired)**: 5 time-domain + MFCC = 87 dims
2. **TD0/TD10 (EMG-UKA corpus)**: Double-moving-average decomposition into
   low-freq articulation vs high-freq muscle firing, 5 features/channel/frame,
   with ±10-frame context stacking. Validated on thousands of EMG speech utterances.

The TD10 pipeline is the default; the legacy pipeline is preserved for
backward compatibility.
"""

import numpy as np
from scipy.fft import rfft


# ══════════════════════════════════════════════════════════════════════════════
# Legacy features (AlterEgo-inspired)
# ══════════════════════════════════════════════════════════════════════════════

# ── Time-domain features ─────────────────────────────────────────────────────

def rms(signal: np.ndarray) -> float:
    """Root Mean Square -- measures signal power."""
    return float(np.sqrt(np.mean(signal ** 2)))


def mean_absolute_value(signal: np.ndarray) -> float:
    """Mean Absolute Value -- amplitude estimate."""
    return float(np.mean(np.abs(signal)))


def waveform_length(signal: np.ndarray) -> float:
    """Waveform Length -- cumulative change, measures signal complexity."""
    return float(np.sum(np.abs(np.diff(signal))))


def zero_crossings(signal: np.ndarray, threshold: float = 0.01) -> int:
    """Count sign changes with a small threshold to avoid noise crossings."""
    shifted = signal - threshold
    sign_changes = np.diff(np.sign(shifted))
    return int(np.sum(np.abs(sign_changes) > 0))


def slope_sign_changes(signal: np.ndarray) -> int:
    """Count changes in slope direction."""
    diff1 = np.diff(signal)
    sign_changes = np.diff(np.sign(diff1))
    return int(np.sum(np.abs(sign_changes) > 0))


def extract_channel_features(signal: np.ndarray) -> list[float]:
    """Extract 5 time-domain features for a single channel."""
    return [
        rms(signal),
        mean_absolute_value(signal),
        waveform_length(signal),
        float(zero_crossings(signal)),
        float(slope_sign_changes(signal)),
    ]


# ── MFCC features (inspired by AlterEgo paper) ──────────────────────────────

def _mel_filterbank(num_filters: int, fft_size: int, sample_rate: float,
                     low_freq: float = 0.0, high_freq: float = None) -> np.ndarray:
    """Create a mel-scale triangular filterbank matrix."""
    if high_freq is None:
        high_freq = sample_rate / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)

    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)

    num_fft_bins = fft_size // 2 + 1
    fbank = np.zeros((num_filters, num_fft_bins))

    for i in range(num_filters):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]

        for j in range(left, center):
            if center > left:
                fbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                fbank[i, j] = (right - j) / (right - center)

    return fbank


def compute_mfcc_channel(signal: np.ndarray, sample_rate: float = 250.0,
                          num_cepstral: int = 8, num_mel_filters: int = 16,
                          frame_size_s: float = 0.025, frame_step_s: float = 0.01
                          ) -> np.ndarray:
    """Compute MFCC features for a single channel.

    Returns:
        1D array: mean + std of each MFCC across frames = 2 * num_cepstral features.
    """
    frame_size = max(int(frame_size_s * sample_rate), 4)
    frame_step = max(int(frame_step_s * sample_rate), 1)

    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)), mode='constant')

    num_frames = 1 + (len(signal) - frame_size) // frame_step
    if num_frames < 1:
        num_frames = 1

    frames = np.zeros((num_frames, frame_size))
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_size
        if end <= len(signal):
            frames[i] = signal[start:end]
        else:
            frames[i, :len(signal) - start] = signal[start:]

    frames *= np.hamming(frame_size)

    fft_size = frame_size
    mag = np.abs(rfft(frames, n=fft_size, axis=1))
    power_spec = (1.0 / fft_size) * mag ** 2

    fbank = _mel_filterbank(num_mel_filters, fft_size, sample_rate,
                             low_freq=0.0, high_freq=sample_rate / 2.0)

    min_cols = min(fbank.shape[1], power_spec.shape[1])
    filter_energies = np.dot(power_spec[:, :min_cols], fbank[:, :min_cols].T)
    filter_energies = np.maximum(filter_energies, 1e-10)

    log_energies = np.log(filter_energies)

    num_filters = log_energies.shape[1]
    n = np.arange(num_filters)
    dct_matrix = np.zeros((num_cepstral, num_filters))
    for k in range(num_cepstral):
        dct_matrix[k] = np.cos(np.pi * k * (2 * n + 1) / (2 * num_filters))

    mfcc = np.dot(log_energies, dct_matrix.T)

    mfcc_mean = np.mean(mfcc, axis=0)
    mfcc_std = np.std(mfcc, axis=0)

    return np.concatenate([mfcc_mean, mfcc_std])


# ── Legacy combined extraction ───────────────────────────────────────────────

def extract_features_legacy(segment: np.ndarray, sample_rate: float = 250.0) -> np.ndarray:
    """Legacy feature extraction (AlterEgo-inspired).

    Combines:
    - Time-domain features: 5 per channel + 3 cross-channel ratios
    - MFCC features: 16 per channel (8 mean + 8 std)

    For 4 channels: (5*4 + 3) + (16*4) = 23 + 64 = 87 features.
    """
    num_channels = segment.shape[1]
    features: list[float] = []

    channel_rms: list[float] = []
    for ch in range(num_channels):
        ch_features = extract_channel_features(segment[:, ch])
        features.extend(ch_features)
        channel_rms.append(ch_features[0])

    eps = 1e-8
    if num_channels >= 4:
        features.append(channel_rms[0] / (channel_rms[1] + eps))
        features.append(channel_rms[2] / (channel_rms[3] + eps))
        features.append(channel_rms[0] / (channel_rms[3] + eps))
    elif num_channels >= 2:
        features.append(channel_rms[0] / (channel_rms[1] + eps))
        features.extend([0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])

    for ch in range(num_channels):
        mfcc_feats = compute_mfcc_channel(segment[:, ch], sample_rate=sample_rate)
        features.extend(mfcc_feats.tolist())

    return np.array(features, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# EMG-UKA TD0/TD10 features
# ══════════════════════════════════════════════════════════════════════════════

def _double_moving_average(x: np.ndarray, window: int = 9) -> np.ndarray:
    """9-point double moving average to extract low-frequency component w[n].

    First pass: forward moving average. Second pass: moving average of the
    result. This approximates the slow articulation movement component.
    """
    kernel = np.ones(window) / window
    # First pass
    w1 = np.convolve(x, kernel, mode='same')
    # Second pass
    w = np.convolve(w1, kernel, mode='same')
    return w


def _compute_td0_channel(signal: np.ndarray, fs: int,
                          frame_size_ms: int = 27,
                          frame_shift_ms: int = 10) -> np.ndarray:
    """Compute 5 TD0 features per frame for a single channel.

    Decomposes signal into:
    - w[n]: smoothed low-freq component (double moving average)
    - p[n] = x[n] - w[n]: high-freq component
    - r[n] = |p[n]|: rectified high-freq

    Per-frame features:
    1. w_bar: frame mean of w[n] (smoothed mean -- articulation position)
    2. P_w:   frame power of w[n] (articulation energy)
    3. P_r:   frame power of r[n] (muscle firing energy)
    4. z_p:   zero-crossing rate of p[n] (muscle firing frequency)
    5. r_bar: frame mean of r[n] (rectified muscle activity)

    Returns:
        (num_frames, 5) array of TD0 features.
    """
    frame_size = max(int(fs * frame_size_ms / 1000), 1)
    frame_shift = max(int(fs * frame_shift_ms / 1000), 1)

    # Pad signal if shorter than one frame
    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)), mode='constant')

    # Decompose
    w = _double_moving_average(signal)
    p = signal - w      # high-freq component
    r = np.abs(p)       # rectified high-freq

    # Frame the signal components
    num_frames = max(1, 1 + (len(signal) - frame_size) // frame_shift)

    td0 = np.zeros((num_frames, 5))
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_size
        if end > len(signal):
            end = len(signal)

        w_frame = w[start:end]
        p_frame = p[start:end]
        r_frame = r[start:end]

        # 1. w_bar: mean of smoothed signal
        td0[i, 0] = np.mean(w_frame)
        # 2. P_w: power of smoothed signal
        td0[i, 1] = np.mean(w_frame ** 2)
        # 3. P_r: power of rectified high-freq
        td0[i, 2] = np.mean(r_frame ** 2)
        # 4. z_p: zero-crossing rate of high-freq
        if len(p_frame) > 1:
            sign_changes = np.abs(np.diff(np.sign(p_frame)))
            td0[i, 3] = np.sum(sign_changes > 0) / len(p_frame)
        # 5. r_bar: mean of rectified high-freq
        td0[i, 4] = np.mean(r_frame)

    return td0


def _stack_context(td0_frames: np.ndarray, context: int = 10) -> np.ndarray:
    """Stack ±context adjacent frames to create TD10 features.

    For each frame t, concatenate frames [t-context, ..., t, ..., t+context].
    Edge frames are padded by repeating the boundary frame.

    Args:
        td0_frames: (num_frames, num_features) TD0 feature matrix.
        context: Number of frames on each side to stack.

    Returns:
        (num_frames, num_features * (2*context + 1)) TD10 feature matrix.
    """
    num_frames, num_feats = td0_frames.shape
    total_width = 2 * context + 1

    # Pad with edge frames
    padded = np.pad(td0_frames, ((context, context), (0, 0)), mode='edge')

    # Stack
    td10 = np.zeros((num_frames, num_feats * total_width))
    for i in range(num_frames):
        td10[i] = padded[i:i + total_width].ravel()

    return td10


def extract_features_td10(segment: np.ndarray, sample_rate: float = 250.0,
                           frame_size_ms: int = 27, frame_shift_ms: int = 10,
                           context: int = 10) -> np.ndarray:
    """Full TD0 -> TD10 pipeline returning frame-level features.

    Args:
        segment: 2D array (num_samples, num_channels).
        sample_rate: Sampling rate in Hz.
        frame_size_ms: Frame size in milliseconds.
        frame_shift_ms: Frame shift in milliseconds.
        context: Number of context frames (±) for stacking.

    Returns:
        2D array (num_frames, num_channels * 5 * (2*context+1)).
        For 4 channels with context=10: (num_frames, 420).
    """
    fs = int(sample_rate)
    num_channels = segment.shape[1]

    # Compute TD0 per channel
    channel_td0s = []
    for ch in range(num_channels):
        td0 = _compute_td0_channel(segment[:, ch], fs,
                                    frame_size_ms=frame_size_ms,
                                    frame_shift_ms=frame_shift_ms)
        channel_td0s.append(td0)

    # Align frame counts (should be the same, but be safe)
    min_frames = min(t.shape[0] for t in channel_td0s)
    channel_td0s = [t[:min_frames] for t in channel_td0s]

    # Concatenate channels: (num_frames, num_channels * 5)
    td0_all = np.concatenate(channel_td0s, axis=1)

    # Stack context: (num_frames, num_channels * 5 * (2*context+1))
    td10 = _stack_context(td0_all, context=context)

    return td10


def extract_features_td10_segment(segment: np.ndarray, sample_rate: float = 250.0,
                                    frame_size_ms: int = 27, frame_shift_ms: int = 10,
                                    context: int = 10) -> np.ndarray:
    """Aggregate frame-level TD10 features into a single segment-level vector.

    Computes mean and std across frames for each TD10 dimension.
    This adapts the EMG-UKA frame-level approach to SilentPilot's
    segment-level classification.

    Returns:
        1D array of length num_channels * 5 * (2*context+1) * 2.
        For 4 channels with context=10: 420 * 2 = 840 dims.
    """
    td10 = extract_features_td10(segment, sample_rate,
                                  frame_size_ms=frame_size_ms,
                                  frame_shift_ms=frame_shift_ms,
                                  context=context)

    frame_mean = np.mean(td10, axis=0)
    frame_std = np.std(td10, axis=0)

    return np.concatenate([frame_mean, frame_std]).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Public API -- defaults to TD10 pipeline
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(segment: np.ndarray, sample_rate: float = 250.0) -> np.ndarray:
    """Extract feature vector from a multi-channel segment.

    Uses the EMG-UKA TD10 pipeline (segment-level aggregation).

    Args:
        segment: 2D array of shape (num_samples, num_channels).
        sample_rate: Sampling rate in Hz.

    Returns:
        1D feature vector.
    """
    return extract_features_td10_segment(segment, sample_rate)


def extract_features_batch(segments: list[np.ndarray],
                            sample_rate: float = 250.0) -> np.ndarray:
    """Extract features for a batch of segments.

    Args:
        segments: List of 2D arrays, each (num_samples, num_channels).
        sample_rate: Sampling rate.

    Returns:
        2D array of shape (num_segments, num_features).
    """
    return np.array([extract_features(seg, sample_rate) for seg in segments])
