#!/usr/bin/env python3
"""Synthetic word composition from phone-level EMG segments.

Idea:
  We have phone-level EMG segments (real data) with 75-1700 exemplars per phone.
  We can look up any English word's phoneme sequence via CMU dict.
  So: compose synthetic "word" EMG by concatenating real phone EMG segments,
  creating unlimited training data for ANY word we choose.

Pipeline:
  1. Build phone exemplar bank from silent EMG data (raw EMG segments per phone)
  2. For each target command word:
     a. Look up phone sequence (e.g., OPEN → OW P AH N)
     b. Pick random exemplars for each phone
     c. Concatenate with cross-fade at boundaries
     d. Resample to fixed word length
     e. Preprocess (bandpass + notch) and extract features
  3. Generate N synthetic samples per command
  4. Evaluate:
     - Synthetic→Synthetic (sanity: is the model self-consistent?)
     - Synthetic→Real (KEY: does synthetic training transfer to real words?)
"""

import os, sys, time, glob, warnings
import numpy as np
from collections import defaultdict, Counter
from scipy.signal import resample, butter, filtfilt, iirnotch

import pronouncing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emg_core.dsp.features import extract_features

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ── Constants ──
CORPUS = os.path.expanduser(
    "~/.cache/kagglehub/datasets/xabierdezuazo/emguka-trial-corpus/versions/1/EMG-UKA-Trial-Corpus"
)
FS = 600            # EMG sample rate
FPF = 6.0           # EMG frames per phone frame (600 Hz / 100 FPS)
NCH = 6             # EMG channels to use
TOTAL_CH = 7        # Total channels in .adc file
WORD_SEG_LEN = 180  # ~300ms fixed segment length for words
CROSSFADE = 6       # Samples to cross-fade between phones (~10ms at 600Hz)

# Pre-compute filter coefficients
_nyq = FS / 2.0
_b_bp, _a_bp = butter(4, [1.3/_nyq, 50.0/_nyq], btype='band')
_b_n, _a_n = iirnotch(60.0, 30.0, FS)

# Our 8 original command words
COMMANDS = ["OPEN", "SEARCH", "CLICK", "SCROLL", "TYPE", "ENTER", "CONFIRM", "CANCEL"]


# ══════════════════════════════════════════════════════════════════════
# Data loading (same as emguka_commands.py)
# ══════════════════════════════════════════════════════════════════════

def load_emg(sp, sess, utt):
    d = os.path.join(CORPUS, "emg", sp, sess)
    m = glob.glob(os.path.join(d, f"*_{sp}_{sess}_{utt}.adc"))
    if not m: return None
    raw = np.fromfile(m[0], dtype=np.int16)
    return raw.reshape(len(raw)//TOTAL_CH, TOTAL_CH)[:, :NCH].astype(np.float64)

def load_align(sp, sess, utt):
    f = os.path.join(CORPUS, "Alignments", sp, sess, f"phones_{sp}_{sess}_{utt}.txt")
    if not os.path.exists(f): return []
    return [(int(p[0]),int(p[1]),p[2]) for ln in open(f) for p in [ln.split()] if len(p)>=3]

def load_transcript(sp, sess, utt):
    f = os.path.join(CORPUS, "Transcripts", sp, sess, f"transcript_{sp}_{sess}_{utt}.txt")
    if not os.path.exists(f): return ""
    return open(f).read().strip()

def parse_subset(fn):
    r = defaultdict(list)
    for line in open(os.path.join(CORPUS, "Subsets", fn)):
        p = line.strip().split(":")
        if len(p)<2: continue
        ids = p[1].strip().split()
        if not ids: continue
        sp, sess = p[0].strip().replace("emg_","").split("-")
        for uid in ids: r[sp].append((sp,sess,uid.split("-")[-1]))
    return r


# ══════════════════════════════════════════════════════════════════════
# Step 1: Build phone exemplar bank
# ══════════════════════════════════════════════════════════════════════

def build_phone_bank(utts, max_per_phone=200):
    """Extract raw EMG segments for each phone from utterances.

    For each non-SIL phone in each utterance:
      - Extract the raw EMG segment (phone frames → EMG samples)
      - Store as a (samples, channels) array

    Returns: {phone_label: [np.array(samples, NCH), ...]}
    """
    bank = defaultdict(list)
    t0 = time.time()

    for i, (sp, sess, utt) in enumerate(utts):
        if (i+1) % 50 == 0:
            P(f"    [{i+1}/{len(utts)}] {time.time()-t0:.1f}s")

        emg = load_emg(sp, sess, utt)
        align = load_align(sp, sess, utt)
        if emg is None or not align:
            continue

        for sf, ef, ph in align:
            if ph == "SIL":
                continue
            if len(bank[ph]) >= max_per_phone:
                continue

            # Convert phone frames to EMG sample indices
            s0 = max(0, int(sf * FPF))
            s1 = min(len(emg), int((ef + 1) * FPF))
            if s1 - s0 < 6:  # too short
                continue

            bank[ph].append(emg[s0:s1].copy())

    elapsed = time.time() - t0
    total = sum(len(v) for v in bank.values())
    P(f"    Phone bank: {len(bank)} phones, {total} exemplars, {elapsed:.1f}s")
    return dict(bank)


# ══════════════════════════════════════════════════════════════════════
# Step 2: Synthesize word-level EMG from phone exemplars
# ══════════════════════════════════════════════════════════════════════

def get_word_phones(word):
    """Get ARPABET phone sequence for a word (stripped of stress)."""
    phones = pronouncing.phones_for_word(word.lower())
    if not phones:
        return None
    return [p.rstrip("012") for p in phones[0].split()]


def crossfade_segments(segs, fade=CROSSFADE):
    """Concatenate EMG segments with linear cross-fade at boundaries.

    This smooths the transition between consecutive phones to reduce
    concatenation artifacts (simulating coarticulation effects).
    """
    if not segs:
        return np.zeros((0, NCH))
    if len(segs) == 1:
        return segs[0]

    result = segs[0].copy()
    for seg in segs[1:]:
        overlap = min(fade, len(result), len(seg))
        if overlap > 0:
            # Linear crossfade in the overlap region
            alpha = np.linspace(1.0, 0.0, overlap)[:, None]
            result[-overlap:] = result[-overlap:] * alpha + seg[:overlap] * (1 - alpha)
            result = np.vstack([result, seg[overlap:]])
        else:
            result = np.vstack([result, seg])

    return result


def synthesize_word(word_phones, phone_bank, rng, augment=True):
    """Create one synthetic word EMG segment by composing phone exemplars.

    Args:
        word_phones: list of ARPABET phone labels (e.g., ['OW', 'P', 'AH', 'N'])
        phone_bank: dict mapping phone labels to lists of EMG segments
        rng: numpy random generator for reproducibility
        augment: whether to apply small augmentations

    Returns:
        np.array of shape (WORD_SEG_LEN, NCH) -- preprocessed EMG segment
    """
    segments = []
    for ph in word_phones:
        exemplars = phone_bank.get(ph)
        if not exemplars:
            return None  # missing phone
        seg = exemplars[rng.integers(0, len(exemplars))].copy()

        if augment:
            # Small amplitude jitter (±10%)
            seg *= (1.0 + rng.uniform(-0.1, 0.1))
            # Small time stretch (resample to ±15% of original length)
            stretch = rng.uniform(0.85, 1.15)
            new_len = max(6, int(len(seg) * stretch))
            seg = resample(seg, new_len, axis=0)

        segments.append(seg)

    # Concatenate with cross-fade
    raw = crossfade_segments(segments)
    if len(raw) < 12:
        return None

    # Resample to fixed word length
    word_seg = resample(raw, WORD_SEG_LEN, axis=0)

    # Preprocess: DC removal + bandpass + notch (per channel)
    processed = np.empty_like(word_seg, dtype=np.float64)
    for c in range(word_seg.shape[1]):
        x = word_seg[:, c].astype(np.float64)
        x -= np.mean(x)
        try:
            x = filtfilt(_b_bp, _a_bp, x)
            x = filtfilt(_b_n, _a_n, x)
        except Exception:
            pass
        processed[:, c] = x

    return processed


def generate_synthetic_dataset(commands, phone_bank, n_per_class=200, seed=42):
    """Generate synthetic EMG feature vectors for a set of command words.

    Returns: X (n_samples, n_features), y (n_samples,), labels list
    """
    rng = np.random.default_rng(seed)

    # Get phone sequences for all commands
    cmd_phones = {}
    for cmd in commands:
        phones = get_word_phones(cmd)
        if phones is None:
            P(f"    WARNING: '{cmd}' not in CMU dict, skipping")
            continue
        # Check all phones available
        missing = [p for p in phones if p not in phone_bank]
        if missing:
            P(f"    WARNING: '{cmd}' needs phones {missing} not in bank, skipping")
            continue
        cmd_phones[cmd] = phones

    if not cmd_phones:
        return None, None, []

    labels = sorted(cmd_phones.keys())
    lm = {w: i for i, w in enumerate(labels)}

    X_list, y_list = [], []
    for cmd in labels:
        phones = cmd_phones[cmd]
        generated = 0
        attempts = 0
        while generated < n_per_class and attempts < n_per_class * 3:
            attempts += 1
            seg = synthesize_word(phones, phone_bank, rng, augment=True)
            if seg is None:
                continue
            feat = extract_features(seg, sample_rate=FS)
            X_list.append(feat)
            y_list.append(lm[cmd])
            generated += 1

        P(f"    {cmd:10s}: generated {generated}/{n_per_class} "
          f"({'+'.join(phones)})")

    X = np.array(X_list)
    y = np.array(y_list, dtype=int)
    return X, y, labels


# ══════════════════════════════════════════════════════════════════════
# Real word extraction (for test set) -- from emguka_commands.py
# ══════════════════════════════════════════════════════════════════════

def word_nphones(w):
    phones = pronouncing.phones_for_word(w.lower())
    return len(phones[0].split()) if phones else None

def extract_real_words(utts, target_words=None, seg_len=WORD_SEG_LEN, ctx_frames=3):
    """Extract real word-level EMG segments (for testing).

    If target_words is None, extracts ALL words found.
    """
    data = defaultdict(list)
    target_set = set(w.upper() for w in target_words) if target_words else None
    t0 = time.time()
    matched = 0

    for i, (sp, sess, utt) in enumerate(utts):
        emg = load_emg(sp, sess, utt)
        align = load_align(sp, sess, utt)
        trans = load_transcript(sp, sess, utt)
        if emg is None or not align or not trans:
            continue

        words = trans.upper().split()
        phone_counts = []
        skip = False
        for w in words:
            pc = word_nphones(w)
            if pc is None: skip = True; break
            phone_counts.append(pc)
        if skip: continue

        flat_phones = [(sf, ef, ph) for sf, ef, ph in align if ph != "SIL"]
        if not flat_phones: continue
        total_expected = sum(phone_counts)
        if total_expected == 0: continue

        matched += 1
        scale = len(flat_phones) / total_expected
        cum = [0]
        for pc in phone_counts:
            cum.append(cum[-1] + pc)

        for wi, w in enumerate(words):
            if target_set and w not in target_set: continue

            si = int(round(cum[wi] * scale))
            ei = int(round(cum[wi+1] * scale))
            si = max(0, min(si, len(flat_phones)-1))
            ei = max(si+1, min(ei, len(flat_phones)))
            word_phones = flat_phones[si:ei]
            if not word_phones: continue

            sf = word_phones[0][0]
            ef = word_phones[-1][1]
            s0 = max(0, int((sf - ctx_frames) * FPF))
            s1 = min(len(emg), int((ef + 1 + ctx_frames) * FPF))
            if s1 - s0 < 12: continue

            seg = resample(emg[s0:s1], seg_len, axis=0)
            processed = np.empty_like(seg, dtype=np.float64)
            for c in range(seg.shape[1]):
                x = seg[:, c].astype(np.float64)
                x -= np.mean(x)
                try:
                    x = filtfilt(_b_bp, _a_bp, x)
                    x = filtfilt(_b_n, _a_n, x)
                except Exception:
                    pass
                processed[:, c] = x

            feat = extract_features(processed, sample_rate=FS)
            data[w].append(feat)

    elapsed = time.time() - t0
    total = sum(len(v) for v in data.values())
    P(f"    Real words: {total} segs, {len(data)} words, "
      f"{matched}/{len(utts)} utts, {elapsed:.1f}s")
    return dict(data)


# ══════════════════════════════════════════════════════════════════════
# Training and evaluation
# ══════════════════════════════════════════════════════════════════════

def train_and_eval(name, X_train, y_train, X_test, y_test, labels):
    P(f"\n  ┌─ {name}")
    P(f"  │ Train: {len(X_train)}, Test: {len(X_test)}, Commands: {len(labels)}")

    if len(X_train) < 10 or len(set(y_train)) < 2:
        P("  └─ Insufficient data!"); return {}

    chance = 1.0 / len(labels)
    trc = Counter(y_train.tolist())
    P(f"  │ Train dist: {dict(sorted((labels[k],v) for k,v in trc.items()))}")

    # ── LogisticRegression ──
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, C=1.0)),
    ])
    lr.fit(X_train, y_train)
    lr_tr = accuracy_score(y_train, lr.predict(X_train))
    lr_te = accuracy_score(y_test, lr.predict(X_test))

    # ── RandomForest (class-balanced) ──
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=42)),
    ])
    rf.fit(X_train, y_train)
    rf_tr = accuracy_score(y_train, rf.predict(X_train))
    rf_te = accuracy_score(y_test, rf.predict(X_test))

    P(f"  │ LogReg: Train={lr_tr:.1%}, Test={lr_te:.1%} ({lr_te/chance:.1f}x)")
    P(f"  │ RF:     Train={rf_tr:.1%}, Test={rf_te:.1%} ({rf_te/chance:.1f}x)")

    # Best model details
    best_name = "RF" if rf_te >= lr_te else "LogReg"
    best_pipe = rf if rf_te >= lr_te else lr
    best_te = max(rf_te, lr_te)

    y_pred = best_pipe.predict(X_test)
    P(f"  │")
    P(f"  │ Best: {best_name} → {best_te:.1%} ({best_te/chance:.1f}x chance)")
    P(f"  │ Per-class:")
    for i, l in enumerate(labels):
        m = y_test == i
        if m.sum() > 0:
            c = (y_pred[m] == i).sum()
            P(f"  │   {l:>10s}: {c}/{m.sum()} ({c/m.sum():.0%})")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
    P(f"  │ Confusion matrix:")
    header = "  │   " + " " * 10 + " ".join(f"{l[:5]:>5s}" for l in labels)
    P(header)
    for i, l in enumerate(labels):
        row = f"  │   {l:>10s} " + " ".join(f"{cm[i,j]:5d}" for j in range(len(labels)))
        P(row)

    P(f"  └─ {best_name}: {best_te:.1%} ({best_te/chance:.1f}x chance)")

    return {'lr_te': lr_te, 'rf_te': rf_te, 'best': best_name,
            'best_acc': best_te, 'chance': chance, 'n_cls': len(labels)}


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    P("=" * 70)
    P("SYNTHETIC WORD COMPOSITION FROM PHONE-LEVEL EMG")
    P("=" * 70)
    P()
    P("Two experiments:")
    P("  A) Syn→Syn with our 8 original commands (sanity check)")
    P("  B) Syn→Real with corpus words (KEY TRANSFER TEST)")
    P()

    # Parse data splits
    tr_sil = parse_subset("train.silent")
    te_sil = parse_subset("test.silent")
    tr_aud = parse_subset("train.audible")

    results = []

    for sp in ["008", "002"]:
        P(f"\n{'=' * 70}")
        P(f"SPEAKER {sp}")
        P(f"{'=' * 70}")

        sp_tr_sil = tr_sil.get(sp, [])
        sp_te_sil = te_sil.get(sp, [])
        sp_tr_aud = tr_aud.get(sp, [])

        if not sp_tr_sil:
            P(f"  No silent train data for speaker {sp}")
            continue

        # ── Step 1: Build phone exemplar bank ──
        P(f"\n  Building phone bank from silent ({len(sp_tr_sil)}) + "
          f"audible ({len(sp_tr_aud)}) train utterances...")
        all_train = sp_tr_sil + sp_tr_aud
        phone_bank = build_phone_bank(all_train, max_per_phone=200)

        # Map common phone variants
        phone_map = {'AH': 'AX', 'ER': 'AXR'}
        for needed, alt in phone_map.items():
            if needed not in phone_bank and alt in phone_bank:
                phone_bank[needed] = phone_bank[alt]
                P(f"    Mapped {needed} ← {alt} ({len(phone_bank[needed])} exemplars)")

        # ══════════════════════════════════════════════════════════════
        # PART A: Original 8 commands — Syn→Syn only
        # ══════════════════════════════════════════════════════════════

        P(f"\n  ── PART A: Original 8 Commands (Synthetic only) ──")

        # Check which commands we can synthesize
        synth_cmds = []
        for cmd in COMMANDS:
            phones = get_word_phones(cmd)
            if phones and all(p in phone_bank for p in phones):
                synth_cmds.append(cmd)
                P(f"    {cmd:10s} → {' '.join(phones)}")
            else:
                missing = [p for p in (phones or []) if p not in phone_bank]
                P(f"    {cmd:10s} → SKIP (missing: {missing})")

        P(f"\n  Generating synthetic train (200/class) for {len(synth_cmds)} commands...")
        X_syn_tr, y_syn_tr, cmd_labels = generate_synthetic_dataset(
            synth_cmds, phone_bank, n_per_class=200, seed=42)
        P(f"  Generating synthetic test (50/class)...")
        X_syn_te, y_syn_te, _ = generate_synthetic_dataset(
            cmd_labels, phone_bank, n_per_class=50, seed=999)

        if X_syn_tr is not None:
            P(f"\n{'─' * 70}")
            P(f"  EXP A: Syn→Syn — {len(cmd_labels)} original commands")
            P(f"{'─' * 70}")
            r1 = train_and_eval(
                f"Sp{sp} Syn→Syn {len(cmd_labels)} cmds",
                X_syn_tr, y_syn_tr, X_syn_te, y_syn_te, cmd_labels)
            results.append((sp, "Syn→Syn (8 cmds)", r1))

        # ══════════════════════════════════════════════════════════════
        # PART B: Corpus words — Syn→Real transfer test
        # ══════════════════════════════════════════════════════════════

        P(f"\n  ── PART B: Corpus Words (Transfer Test) ──")

        # First, find which words actually exist in the corpus with enough data
        P(f"  Finding words in corpus with sufficient real data...")
        real_train_all = extract_real_words(sp_tr_sil, None)  # extract ALL words
        real_test_all = extract_real_words(sp_te_sil, None)

        # Find words with 5+ train and 2+ test samples that we CAN synthesize
        corpus_words = []
        for w in sorted(real_train_all.keys()):
            tr_n = len(real_train_all[w])
            te_n = len(real_test_all.get(w, []))
            phones = get_word_phones(w)
            can_synth = phones and all(p in phone_bank for p in phones)
            if tr_n >= 5 and te_n >= 2 and can_synth:
                corpus_words.append(w)
                P(f"    {w:12s}: train={tr_n:3d} test={te_n:2d} "
                  f"phones={'+'.join(phones)}")

        if len(corpus_words) < 3:
            P(f"  Not enough synthesizable corpus words!")
            continue

        # Limit to top 8 by sample count
        corpus_words = sorted(corpus_words,
            key=lambda w: len(real_train_all[w]), reverse=True)[:8]
        P(f"\n  Selected {len(corpus_words)} corpus words: {corpus_words}")

        # Generate synthetic data for corpus words
        P(f"\n  Generating synthetic train (200/class)...")
        X_syn_corpus, y_syn_corpus, corpus_labels = generate_synthetic_dataset(
            corpus_words, phone_bank, n_per_class=200, seed=42)

        # Build real train/test matrices for these words
        lm = {w: i for i, w in enumerate(corpus_labels)}

        X_real_tr, y_real_tr = [], []
        for w in corpus_labels:
            for feat in real_train_all.get(w, []):
                X_real_tr.append(feat); y_real_tr.append(lm[w])
        X_real_tr = np.array(X_real_tr) if X_real_tr else np.zeros((0, X_syn_corpus.shape[1]))
        y_real_tr = np.array(y_real_tr, dtype=int)

        X_real_te, y_real_te = [], []
        for w in corpus_labels:
            for feat in real_test_all.get(w, []):
                X_real_te.append(feat); y_real_te.append(lm[w])
        X_real_te = np.array(X_real_te) if X_real_te else np.zeros((0, X_syn_corpus.shape[1]))
        y_real_te = np.array(y_real_te, dtype=int)

        P(f"  Real data: train={len(X_real_tr)}, test={len(X_real_te)}")

        # --- Experiment B1: Synthetic → Real (KEY) ---
        P(f"\n{'─' * 70}")
        P(f"  EXP B1: Syn→Real — {len(corpus_labels)} corpus words")
        P(f"{'─' * 70}")
        if len(X_real_te) > 0:
            r2 = train_and_eval(
                f"Sp{sp} Syn→Real {len(corpus_labels)} words",
                X_syn_corpus, y_syn_corpus, X_real_te, y_real_te, corpus_labels)
            results.append((sp, "Syn→Real", r2))
        else:
            P("  No real test data!")

        # --- Experiment B2: Real → Real (baseline) ---
        P(f"\n{'─' * 70}")
        P(f"  EXP B2: Real→Real — {len(corpus_labels)} corpus words (baseline)")
        P(f"{'─' * 70}")
        if len(X_real_tr) >= 10 and len(X_real_te) > 0:
            r3 = train_and_eval(
                f"Sp{sp} Real→Real {len(corpus_labels)} words",
                X_real_tr, y_real_tr, X_real_te, y_real_te, corpus_labels)
            results.append((sp, "Real→Real", r3))
        else:
            P(f"  Not enough real data")

        # --- Experiment B3: Syn+Real → Real (combined) ---
        P(f"\n{'─' * 70}")
        P(f"  EXP B3: Syn+Real→Real — {len(corpus_labels)} corpus words")
        P(f"{'─' * 70}")
        if len(X_real_tr) >= 5 and len(X_real_te) > 0:
            X_comb = np.vstack([X_syn_corpus, X_real_tr])
            y_comb = np.concatenate([y_syn_corpus, y_real_tr])
            r4 = train_and_eval(
                f"Sp{sp} Syn+Real→Real {len(corpus_labels)} words",
                X_comb, y_comb, X_real_te, y_real_te, corpus_labels)
            results.append((sp, "Syn+Real→Real", r4))

    # ══════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════

    P(f"\n{'=' * 70}")
    P("FINAL SUMMARY")
    P(f"{'=' * 70}")

    for sp, exp, r in results:
        if not r: continue
        P(f"  Sp{sp} {exp:25s}: {r['best']} {r['best_acc']:.1%} "
          f"({r['best_acc']/r['chance']:.1f}x chance) "
          f"[LR={r['lr_te']:.1%}, RF={r['rf_te']:.1%}]")

    P(f"\n{'=' * 70}")
    P("Key comparisons:")
    P("  - Syn→Real vs Real→Real: Does synthetic data transfer to real?")
    P("  - Syn+Real→Real vs Real→Real: Does synthetic data help?")
    P("  - Syn→Syn with 8 cmds: Can we discriminate our custom commands?")
    P(f"{'=' * 70}")


if __name__ == "__main__":
    main()
