#!/usr/bin/env python3
"""EMG-UKA Trial Corpus pipeline: load real EMG data, extract word segments, classify.

This connects the EMG-UKA dataset (real sEMG silent speech data) with our
SilentPilot feature extraction and classification pipeline.

Key insights from iteration 1:
- Must train per-speaker (AlterEgo paper also does this)
- Need sufficient samples per class (>=10)
- MFCC features + time-domain features from our pipeline
- StandardScaler + regularized logistic regression

Dataset: https://www.kaggle.com/datasets/xabierdezuazo/emguka-trial-corpus
Format: 6 EMG channels at 600 Hz, phone-level alignments, full-sentence transcripts.
"""

import os
import sys
import glob
import numpy as np
from collections import Counter, defaultdict
from scipy.signal import resample

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emg_core.dsp.features import extract_features
from emg_core.dsp.filters import preprocess_multichannel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA

# ── Constants ──────────────────────────────────────────────────────────────────

CORPUS_PATH = os.path.expanduser(
    "~/.cache/kagglehub/datasets/xabierdezuazo/emguka-trial-corpus/versions/1/EMG-UKA-Trial-Corpus"
)
EMG_SAMPLE_RATE = 600       # Hz (EMG-UKA standard)
ALIGNMENT_FPS = 100         # Phone alignments at 100 frames/sec
SAMPLES_PER_FRAME = EMG_SAMPLE_RATE / ALIGNMENT_FPS  # = 6.0
NUM_EMG_CHANNELS = 6        # ch0-ch5 are EMG, ch6 is trigger
TOTAL_CHANNELS = 7          # in the .adc file
FIXED_SEGMENT_LENGTH = 180  # samples (~0.3s at 600 Hz) for word-level segments
MIN_WORD_FRAMES = 3         # minimum phone frames to consider a word

# ── Data Loading ──────────────────────────────────────────────────────────────


def load_emg(speaker: str, session: str, utterance: str) -> np.ndarray:
    """Load EMG data from an .adc file. Returns (num_samples, 6)."""
    emg_dir = os.path.join(CORPUS_PATH, "emg", speaker, session)
    pattern = f"*_{speaker}_{session}_{utterance}.adc"
    matches = glob.glob(os.path.join(emg_dir, pattern))
    if not matches:
        return None
    raw = np.fromfile(matches[0], dtype=np.int16)
    n_samples = len(raw) // TOTAL_CHANNELS
    data = raw.reshape(n_samples, TOTAL_CHANNELS)
    return data[:, :NUM_EMG_CHANNELS].astype(np.float64)


def load_alignment(speaker: str, session: str, utterance: str) -> list:
    """Load phone-level alignment. Returns list of (start, end, phone)."""
    align_dir = os.path.join(CORPUS_PATH, "Alignments", speaker, session)
    fname = f"phones_{speaker}_{session}_{utterance}.txt"
    fpath = os.path.join(align_dir, fname)
    if not os.path.exists(fpath):
        return []
    alignment = []
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                alignment.append((int(parts[0]), int(parts[1]), parts[2]))
    return alignment


def load_transcript(speaker: str, session: str, utterance: str) -> str:
    """Load the transcript for an utterance."""
    trans_dir = os.path.join(CORPUS_PATH, "Transcripts", speaker, session)
    fname = f"transcript_{speaker}_{session}_{utterance}.txt"
    fpath = os.path.join(trans_dir, fname)
    if not os.path.exists(fpath):
        return ""
    return open(fpath).read().strip()


def parse_subset_file(filename: str) -> dict:
    """Parse subset file. Returns dict[session_key] -> list of (sp, sess, utt)."""
    fpath = os.path.join(CORPUS_PATH, "Subsets", filename)
    result = defaultdict(list)
    with open(fpath) as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) < 2:
                continue
            session_key = parts[0].strip()
            utt_ids = parts[1].strip().split()
            if not utt_ids:
                continue
            sp, sess = session_key.replace("emg_", "").split("-")
            for utt_id in utt_ids:
                utt_num = utt_id.split("-")[-1]
                result[session_key].append((sp, sess, utt_num))
    return result


# ── Word Extraction ───────────────────────────────────────────────────────────


def extract_word_segments(emg: np.ndarray, alignment: list, transcript: str
                          ) -> list[tuple[str, np.ndarray]]:
    """Extract word-level EMG segments using phone alignment and transcript."""
    if emg is None or not alignment or not transcript:
        return []

    words = transcript.split()

    # Group phones into word-like segments (between SIL boundaries)
    phone_groups = []
    current_group = []
    for start_f, end_f, phone in alignment:
        if phone == "SIL":
            if current_group:
                phone_groups.append(current_group)
                current_group = []
        else:
            current_group.append((start_f, end_f, phone))
    if current_group:
        phone_groups.append(current_group)

    word_segments = []
    for group, word in zip(phone_groups, words):
        if not group:
            continue
        start_sample = int(group[0][0] * SAMPLES_PER_FRAME)
        end_sample = int((group[-1][1] + 1) * SAMPLES_PER_FRAME)
        n_frames = group[-1][1] - group[0][0] + 1
        if n_frames < MIN_WORD_FRAMES:
            continue
        start_sample = max(0, start_sample)
        end_sample = min(len(emg), end_sample)
        if end_sample <= start_sample + 6:
            continue
        segment = emg[start_sample:end_sample]
        word_segments.append((word, segment))

    return word_segments


# ── Feature Extraction ────────────────────────────────────────────────────────


def preprocess_and_extract(segment: np.ndarray) -> np.ndarray:
    """Preprocess EMG segment and extract features."""
    if len(segment) < 6:
        return None
    resampled = resample(segment, FIXED_SEGMENT_LENGTH, axis=0)
    processed = preprocess_multichannel(
        resampled, fs=EMG_SAMPLE_RATE, apply_bandpass=True, apply_notch=True
    )
    features = extract_features(processed, sample_rate=EMG_SAMPLE_RATE)
    return features


# ── Dataset Building ──────────────────────────────────────────────────────────


def collect_word_features(utterances: list) -> dict:
    """Collect word features from utterances. Returns word -> list of feature vectors."""
    word_data = defaultdict(list)
    loaded = 0
    for sp, sess, utt_num in utterances:
        emg = load_emg(sp, sess, utt_num)
        alignment = load_alignment(sp, sess, utt_num)
        transcript = load_transcript(sp, sess, utt_num)
        if emg is None or not alignment:
            continue
        loaded += 1
        for word, seg in extract_word_segments(emg, alignment, transcript):
            features = preprocess_and_extract(seg)
            if features is not None:
                word_data[word].append(features)
    return dict(word_data), loaded


def build_dataset(word_data: dict, vocabulary: list) -> tuple:
    """Build X, y from word_data using given vocabulary."""
    label_to_idx = {w: i for i, w in enumerate(vocabulary)}
    X_list, y_list = [], []
    for word in vocabulary:
        if word in word_data:
            for feat in word_data[word]:
                X_list.append(feat)
                y_list.append(label_to_idx[word])
    if not X_list:
        return np.zeros((0, 129)), np.zeros(0, dtype=int)
    return np.array(X_list), np.array(y_list)


# ── Experiments ───────────────────────────────────────────────────────────────


def run_experiment(name: str, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray, labels: list):
    """Run a classification experiment and print results."""
    print(f"\n{'─'*60}")
    print(f"  Experiment: {name}")
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples, "
          f"Classes: {len(labels)}")
    print(f"{'─'*60}")

    if len(X_train) == 0 or len(set(y_train)) < 2:
        print("  Insufficient training data!")
        return 0.0

    # Pipeline: StandardScaler + LogisticRegression
    pipe_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, C=0.1, solver='lbfgs')),
    ])

    # Pipeline: StandardScaler + PCA + LogisticRegression
    n_components = min(30, X_train.shape[1], X_train.shape[0] - 1)
    pipe_pca = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('clf', LogisticRegression(max_iter=5000, C=0.1, solver='lbfgs')),
    ])

    # Pipeline: StandardScaler + RandomForest
    pipe_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                        random_state=42)),
    ])

    results = {}
    for pipe_name, pipe in [("LR", pipe_lr), ("PCA+LR", pipe_pca), ("RF", pipe_rf)]:
        pipe.fit(X_train, y_train)

        # Cross-val
        try:
            min_class_count = min(Counter(y_train).values())
            n_folds = min(5, min_class_count)
            if n_folds >= 2:
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                             scoring='accuracy')
                cv_str = f"{cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})"
            else:
                cv_str = "N/A (too few per class)"
        except Exception:
            cv_str = "N/A"

        train_acc = accuracy_score(y_train, pipe.predict(X_train))

        if len(X_test) > 0:
            y_pred = pipe.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
        else:
            test_acc = 0.0

        results[pipe_name] = (train_acc, test_acc, cv_str)
        print(f"\n  {pipe_name:>8s}: Train {train_acc:.1%}, Test {test_acc:.1%}, "
              f"CV {cv_str}")

    # Detailed report for best model
    best_name = max(results, key=lambda k: results[k][1])
    print(f"\n  Best model: {best_name}")

    if len(X_test) > 0:
        # Re-fit best and show per-class
        if best_name == "LR":
            pipe = pipe_lr
        elif best_name == "PCA+LR":
            pipe = pipe_pca
        else:
            pipe = pipe_rf
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Determine which classes appear in test
        test_classes = sorted(set(y_test) | set(y_pred))
        test_labels = [labels[i] for i in test_classes]

        print(f"\n  Per-class test results:")
        for cls_idx in test_classes:
            mask = y_test == cls_idx
            if mask.sum() > 0:
                correct = (y_pred[mask] == cls_idx).sum()
                total = mask.sum()
                print(f"    {labels[cls_idx]:>15s}: {correct}/{total} "
                      f"({correct/total:.0%})")

    best_test = results[best_name][1]
    return best_test


def main():
    print("=" * 70)
    print("EMG-UKA Silent Speech -> Word Classification Pipeline")
    print("=" * 70)

    # ── Parse subsets ──
    train_subsets = parse_subset_file("train.silent")
    test_subsets = parse_subset_file("test.silent")

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Per-speaker word classification (Speaker 008)
    # This matches the AlterEgo paper's per-user approach
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Per-Speaker (008) Word Classification")
    print("=" * 70)

    # Speaker 008 has sessions 002 and 003 in silent mode
    sp008_train = []
    sp008_test = []
    for key in train_subsets:
        if "008" in key:
            sp008_train.extend(train_subsets[key])
    for key in test_subsets:
        if "008" in key:
            sp008_test.extend(test_subsets[key])

    print(f"\nSpeaker 008: {len(sp008_train)} train, {len(sp008_test)} test utterances")

    print("\nCollecting train word features...")
    train_wd, n_loaded = collect_word_features(sp008_train)
    print(f"  Loaded {n_loaded} utterances, {len(train_wd)} unique words")

    print("Collecting test word features...")
    test_wd, n_loaded_t = collect_word_features(sp008_test)
    print(f"  Loaded {n_loaded_t} utterances, {len(test_wd)} unique words")

    # Select vocabulary: words with enough training samples
    min_train = 8
    vocab_008 = sorted([w for w, feats in train_wd.items() if len(feats) >= min_train])
    print(f"\nVocabulary (>={min_train} train samples): {len(vocab_008)} words")
    for w in vocab_008[:20]:
        n_tr = len(train_wd.get(w, []))
        n_te = len(test_wd.get(w, []))
        print(f"  {w:>15s}: {n_tr} train, {n_te} test")

    X_tr, y_tr = build_dataset(train_wd, vocab_008)
    X_te, y_te = build_dataset(test_wd, vocab_008)

    if len(X_tr) > 0:
        run_experiment("Speaker 008 Words", X_tr, y_tr, X_te, y_te, vocab_008)

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Per-speaker with broader vocabulary (all speakers)
    # Train and test on same speaker, trying each speaker
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Per-Speaker Word Classification (All Speakers)")
    print("=" * 70)

    speakers = set()
    for key in train_subsets:
        sp = key.replace("emg_", "").split("-")[0]
        speakers.add(sp)

    all_results = []
    for sp in sorted(speakers):
        sp_train = []
        sp_test = []
        for key in train_subsets:
            if sp in key:
                sp_train.extend(train_subsets[key])
        for key in test_subsets:
            if sp in key:
                sp_test.extend(test_subsets[key])

        if not sp_train:
            continue

        print(f"\n  Speaker {sp}: {len(sp_train)} train, {len(sp_test)} test utterances")
        tr_wd, _ = collect_word_features(sp_train)
        te_wd, _ = collect_word_features(sp_test)

        vocab = sorted([w for w, feats in tr_wd.items() if len(feats) >= 5])
        if len(vocab) < 3:
            print(f"  Too few classes for speaker {sp}")
            continue

        X_tr_sp, y_tr_sp = build_dataset(tr_wd, vocab)
        X_te_sp, y_te_sp = build_dataset(te_wd, vocab)

        if len(X_tr_sp) > 0 and len(set(y_tr_sp)) >= 2:
            test_acc = run_experiment(
                f"Speaker {sp}", X_tr_sp, y_tr_sp, X_te_sp, y_te_sp, vocab
            )
            all_results.append((sp, test_acc, len(vocab)))

    # ═══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Sentence-level classification (easier task)
    # Classify which sentence was spoken (higher expected accuracy)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Sentence-Level Classification (Speaker 008)")
    print("=" * 70)

    # Use whole utterance features instead of word-level
    print("\nExtracting whole-utterance features...")
    sent_train_data = defaultdict(list)
    for sp, sess, utt_num in sp008_train:
        emg = load_emg(sp, sess, utt_num)
        transcript = load_transcript(sp, sess, utt_num)
        if emg is None or not transcript:
            continue
        # Use full utterance, resample to fixed length
        fixed_len = 600  # 1 second at 600 Hz (longer for sentences)
        if len(emg) < 12:
            continue
        resampled = resample(emg, fixed_len, axis=0)
        processed = preprocess_multichannel(resampled, fs=EMG_SAMPLE_RATE,
                                             apply_bandpass=True, apply_notch=True)
        features = extract_features(processed, sample_rate=EMG_SAMPLE_RATE)
        sent_train_data[transcript].append(features)

    sent_test_data = defaultdict(list)
    for sp, sess, utt_num in sp008_test:
        emg = load_emg(sp, sess, utt_num)
        transcript = load_transcript(sp, sess, utt_num)
        if emg is None or not transcript:
            continue
        if len(emg) < 12:
            continue
        resampled = resample(emg, 600, axis=0)
        processed = preprocess_multichannel(resampled, fs=EMG_SAMPLE_RATE,
                                             apply_bandpass=True, apply_notch=True)
        features = extract_features(processed, sample_rate=EMG_SAMPLE_RATE)
        sent_test_data[transcript].append(features)

    # Select sentences with enough samples
    sent_vocab = sorted([s for s, feats in sent_train_data.items()
                         if len(feats) >= 3])
    print(f"\nSentences with >=3 train samples: {len(sent_vocab)}")
    for s in sent_vocab[:10]:
        n_tr = len(sent_train_data[s])
        n_te = len(sent_test_data.get(s, []))
        print(f"  [{n_tr} tr, {n_te} te] {s[:60]}")

    if sent_vocab:
        X_tr_s, y_tr_s = build_dataset(sent_train_data, sent_vocab)
        X_te_s, y_te_s = build_dataset(sent_test_data, sent_vocab)
        # Use short labels for display
        short_labels = [s[:25] + "..." if len(s) > 28 else s for s in sent_vocab]
        run_experiment("Sentence Classification", X_tr_s, y_tr_s,
                       X_te_s, y_te_s, short_labels)

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if all_results:
        print("\nPer-speaker word classification results:")
        for sp, acc, n_classes in all_results:
            print(f"  Speaker {sp}: {acc:.1%} test accuracy ({n_classes} word classes)")

    chance_word = 1.0 / max(len(vocab_008), 1) if vocab_008 else 0
    print(f"\n  Chance level (word): {chance_word:.1%}")
    if sent_vocab:
        chance_sent = 1.0 / len(sent_vocab)
        print(f"  Chance level (sentence): {chance_sent:.1%}")

    print(f"\n{'='*70}")
    print("Pipeline complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
