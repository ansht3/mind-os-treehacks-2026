#!/usr/bin/env python3
"""Full stress test: all 8 commands, varied utterance lengths, noise levels.

Tests robustness of the pipeline beyond the basic e2e_test.
"""

import asyncio
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emg_core.ingest.mock_reader import MockReader
from emg_core.dsp.segment import PTTSegmenter
from emg_core.ml.train import train_model
from emg_core.ml.infer import InferenceEngine
from emg_core import config


async def collect_varied_samples(reader: MockReader, command: str, n_samples: int):
    """Collect samples with varied utterance lengths (60-140 raw samples)."""
    rng = np.random.default_rng()
    segmenter = PTTSegmenter()
    segments = []

    for i in range(n_samples):
        # Vary utterance length: 60-140 samples (0.24s - 0.56s at 250Hz)
        utterance_len = rng.integers(60, 141)

        reader.start_utterance(command)
        segmenter.start(label=command)

        for _ in range(utterance_len):
            sample = await reader.read()
            segmenter.add_sample([float(v) for v in sample.ch])

        reader.stop_utterance()
        segment = segmenter.stop()

        if segment:
            segments.append({
                "samples": np.array(segment.samples),
                "label": command,
            })

        # Inter-trial gap (varied)
        gap = rng.integers(15, 40)
        for _ in range(gap):
            await reader.read()

    return segments


async def main():
    print("=" * 60)
    print("SilentPilot FULL Stress Test (All 8 Commands)")
    print("=" * 60)

    user_id = "stress_test"
    commands = ["OPEN", "SEARCH", "CLICK", "SCROLL", "TYPE", "ENTER", "CONFIRM", "CANCEL"]
    n_train = 30   # per class
    n_test = 15    # per class

    # ── 1. Collect calibration data ──
    print(f"\n[1] Collecting calibration data ({n_train} per class, 8 classes)...")
    reader = MockReader()
    await reader.connect()

    all_segments = []
    all_labels = []

    for cmd in commands:
        samples = await collect_varied_samples(reader, cmd, n_train)
        print(f"  {cmd:>8s}: {len(samples)} segments")
        for s in samples:
            all_segments.append(s["samples"])
            all_labels.append(s["label"])

    await reader.disconnect()

    os.makedirs(config.DATA_DIR, exist_ok=True)
    path = os.path.join(config.DATA_DIR, f"{user_id}_calib.npz")
    np.savez(path, segments=np.array(all_segments, dtype=object),
             labels=np.array(all_labels))
    print(f"  Total: {len(all_segments)} segments")

    # ── 2. Train ──
    print("\n[2] Training model...")
    result = train_model(user_id)
    print(f"  Train/val accuracy: {result.accuracy:.1%}")
    print(f"  Per-class:")
    for label, acc in sorted(result.per_class_accuracy.items()):
        print(f"    {label:>8s}: {acc:.0%}")
    print(f"  Confusion matrix:")
    for i, label in enumerate(result.labels):
        row = result.confusion_matrix[i]
        print(f"    {label:>8s}: {row}")

    # ── 3. Live inference ──
    print(f"\n[3] Live inference ({n_test} per class, fresh reader)...")
    engine = InferenceEngine(user_id, confidence_threshold=0.5, cooldown_ms=0)

    reader2 = MockReader()
    await reader2.connect()
    segmenter = PTTSegmenter()
    rng = np.random.default_rng()

    correct = 0
    total = 0
    per_class_correct: dict[str, int] = {c: 0 for c in commands}
    per_class_total: dict[str, int] = {c: 0 for c in commands}

    for cmd in commands:
        for trial in range(n_test):
            # Vary utterance length
            utt_len = rng.integers(60, 141)

            reader2.start_utterance(cmd)
            segmenter.start(label=cmd)

            for _ in range(utt_len):
                sample = await reader2.read()
                segmenter.add_sample([float(v) for v in sample.ch])

            reader2.stop_utterance()
            segment = segmenter.stop()

            if segment:
                seg_array = np.array(segment.samples, dtype=np.float64)
                pred_cmd, pred_conf, _ = engine.predict_raw(seg_array)

                total += 1
                per_class_total[cmd] += 1
                if pred_cmd == cmd:
                    correct += 1
                    per_class_correct[cmd] += 1
                elif trial < 3:
                    print(f"  MISS: {cmd} -> {pred_cmd}({pred_conf:.2f})")

            for _ in range(rng.integers(15, 35)):
                await reader2.read()

    await reader2.disconnect()

    # ── 4. Results ──
    print(f"\n{'='*60}")
    print(f"RESULTS (8 commands)")
    print(f"{'='*60}")
    acc_pct = correct / max(total, 1)
    print(f"\n  Overall live accuracy: {correct}/{total} ({acc_pct:.1%})")
    print(f"\n  Per-class:")
    for cmd in commands:
        c = per_class_correct[cmd]
        t = per_class_total[cmd]
        print(f"    {cmd:>8s}: {c}/{t} ({c/max(t,1):.0%})")

    # Cleanup
    os.remove(path)
    model_path = os.path.join(config.MODELS_DIR, f"{user_id}_model.joblib")
    if os.path.exists(model_path):
        os.remove(model_path)

    print(f"\n{'='*60}")
    target = "PASS" if acc_pct >= 0.90 else "FAIL"
    print(f"  Target >= 90%: {target}")
    print(f"{'='*60}")

    return acc_pct


if __name__ == "__main__":
    accuracy = asyncio.run(main())
    sys.exit(0 if accuracy >= 0.90 else 1)
