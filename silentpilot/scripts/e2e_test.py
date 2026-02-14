#!/usr/bin/env python3
"""End-to-end test: mock EMG -> calibrate -> train -> infer.

Runs without the server -- tests the full pipeline programmatically.
Follows AlterEgo paper methodology:
- Collect N samples per class (paper used 75, we use 30 for speed)
- Train/test split evaluation
- Live inference evaluation with fresh data
"""

import asyncio
import os
import sys
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emg_core.ingest.mock_reader import MockReader
from emg_core.dsp.segment import PTTSegmenter
from emg_core.dsp.features import extract_features
from emg_core.dsp.filters import preprocess_multichannel
from emg_core.ml.train import train_model
from emg_core.ml.infer import InferenceEngine
from emg_core import config


async def collect_samples(reader: MockReader, command: str, n_samples: int,
                           utterance_samples: int = 100):
    """Collect calibration samples for a command using the mock reader.

    Args:
        reader: MockReader instance (must be connected).
        command: Command label to simulate.
        n_samples: Number of segments to collect.
        utterance_samples: Number of raw samples per utterance (~0.4s at 250Hz).
    """
    segmenter = PTTSegmenter()
    segments = []

    for i in range(n_samples):
        # Simulate PTT press
        reader.start_utterance(command)
        segmenter.start(label=command)

        # Collect utterance data
        for _ in range(utterance_samples):
            sample = await reader.read()
            segmenter.add_sample([float(v) for v in sample.ch])

        # Simulate PTT release
        reader.stop_utterance()
        segment = segmenter.stop()

        if segment:
            segments.append({
                "samples": np.array(segment.samples),
                "label": command,
            })

        # Brief pause between samples (clear any lingering state)
        for _ in range(25):
            await reader.read()

    return segments


async def main():
    print("=" * 60)
    print("SilentPilot E2E Test (Paper-Inspired)")
    print("=" * 60)

    user_id = "e2e_test"
    commands = ["OPEN", "SEARCH", "CLICK", "SCROLL"]
    n_train_samples = 30   # per class (paper used 75)
    n_test_samples = 10    # per class for live eval

    # ── 1. Collect calibration data ──
    print(f"\n[1] Collecting calibration data ({n_train_samples} per class)...")
    reader = MockReader()
    await reader.connect()

    all_segments = []
    all_labels = []

    for cmd in commands:
        samples = await collect_samples(reader, cmd, n_samples=n_train_samples)
        print(f"  {cmd}: collected {len(samples)} segments")
        for s in samples:
            all_segments.append(s["samples"])
            all_labels.append(s["label"])

    await reader.disconnect()

    # Save calibration data
    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, f"{user_id}_calib.npz")
    np.savez(
        path,
        segments=np.array(all_segments, dtype=object),
        labels=np.array(all_labels),
    )
    print(f"  Saved {len(all_segments)} total segments to {path}")

    # ── 2. Train model ──
    print("\n[2] Training model...")
    result = train_model(user_id)
    print(f"  Train/val accuracy: {result.accuracy:.1%}")
    print(f"  Per-class: {result.per_class_accuracy}")
    print(f"  Labels: {result.labels}")
    print(f"  Feature count: {len(all_segments[0][0]) if all_segments else 'N/A'} samples -> "
          f"features extracted")
    print(f"  Confusion matrix:")
    for i, label in enumerate(result.labels):
        row = result.confusion_matrix[i]
        print(f"    {label:>8s}: {row}")

    # ── 3. Live inference test ──
    print(f"\n[3] Testing live inference ({n_test_samples} per class)...")
    engine = InferenceEngine(user_id, confidence_threshold=0.5, cooldown_ms=0)

    # Create a FRESH reader with different random state
    reader2 = MockReader()
    await reader2.connect()
    segmenter = PTTSegmenter()

    correct = 0
    total = 0
    per_class_correct: dict[str, int] = {cmd: 0 for cmd in commands}
    per_class_total: dict[str, int] = {cmd: 0 for cmd in commands}
    confusion: dict[str, dict[str, int]] = {cmd: {c: 0 for c in commands} for cmd in commands}

    for cmd in commands:
        for trial in range(n_test_samples):
            reader2.start_utterance(cmd)
            segmenter.start(label=cmd)

            for _ in range(100):
                sample = await reader2.read()
                segmenter.add_sample([float(v) for v in sample.ch])

            reader2.stop_utterance()
            segment = segmenter.stop()

            if segment:
                seg_array = np.array(segment.samples, dtype=np.float64)
                # Use predict_raw to bypass cooldown
                pred_cmd, pred_conf, all_proba = engine.predict_raw(seg_array)

                total += 1
                per_class_total[cmd] += 1
                if pred_cmd == cmd:
                    correct += 1
                    per_class_correct[cmd] += 1
                confusion[cmd][pred_cmd] += 1

                status = "OK" if pred_cmd == cmd else "MISS"
                if trial < 3 or status == "MISS":  # Print first 3 + all misses
                    print(f"  {cmd} -> {pred_cmd}({pred_conf:.2f}) [{status}]")

            # Clear state between trials
            for _ in range(25):
                await reader2.read()

    await reader2.disconnect()

    # ── 4. Results ──
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"\n  Overall live accuracy: {correct}/{total} ({correct/max(total,1):.1%})")
    print(f"\n  Per-class accuracy:")
    for cmd in commands:
        c = per_class_correct[cmd]
        t = per_class_total[cmd]
        print(f"    {cmd:>8s}: {c}/{t} ({c/max(t,1):.0%})")

    print(f"\n  Live confusion matrix:")
    header = "         " + " ".join(f"{c:>8s}" for c in commands)
    print(header)
    for cmd in commands:
        row = " ".join(f"{confusion[cmd][c]:>8d}" for c in commands)
        print(f"    {cmd:>8s} {row}")

    # ── Cleanup ──
    os.remove(path)
    model_path = os.path.join(config.MODELS_DIR, f"{user_id}_model.joblib")
    if os.path.exists(model_path):
        os.remove(model_path)

    print(f"\n{'='*60}")
    print(f"E2E Test Complete!")
    print(f"{'='*60}")

    return correct / max(total, 1)


if __name__ == "__main__":
    accuracy = asyncio.run(main())
    sys.exit(0 if accuracy >= 0.8 else 1)
