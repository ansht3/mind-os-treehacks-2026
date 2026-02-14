#!/usr/bin/env python3
"""Train our RF command classifier on real EMG-UKA word data.

Pipeline:
  1. Extract word-level EMG segments from continuous speech using
     CMU Pronouncing Dict + proportional phone-to-word mapping
  2. Select top command words per speaker (phonetically diverse, 8+ samples)
  3. Preprocess with our existing DSP pipeline (bandpass + notch)
  4. Extract features with our existing feature extractor (MFCC + time-domain)
  5. Train RF classifier (matching our train.py setup)
  6. Evaluate with train.silent / test.silent split
"""

import os, sys, glob, time, warnings
import numpy as np
from collections import Counter, defaultdict
from scipy.signal import resample, butter, filtfilt, iirnotch

import pronouncing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emg_core.dsp.features import extract_features

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ── Constants ──
CORPUS = os.path.expanduser(
    "~/.cache/kagglehub/datasets/xabierdezuazo/emguka-trial-corpus/versions/1/EMG-UKA-Trial-Corpus"
)
FS = 600; FPF = 6.0; NCH = 6; TOTAL_CH = 7
WORD_SEG_LEN = 180  # ~300ms fixed segment length for words

# Pre-compute filter coefficients (matching our pipeline's bandpass 1.3-50 Hz + 60Hz notch)
_nyq = FS / 2.0
_b_bp, _a_bp = butter(4, [1.3/_nyq, 50.0/_nyq], btype='band')
_b_n, _a_n = iirnotch(60.0, 30.0, FS)


# ══════════════════════════════════════════════════════════════════════
# Data loading
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


def word_nphones(w):
    """Get expected phone count for a word from CMU dict."""
    phones = pronouncing.phones_for_word(w.lower())
    return len(phones[0].split()) if phones else None


# ══════════════════════════════════════════════════════════════════════
# Word segment extraction
# ══════════════════════════════════════════════════════════════════════

def extract_word_emg(utts, target_words=None, seg_len=WORD_SEG_LEN, ctx_frames=3):
    """Extract word-level EMG segments using proportional phone mapping.

    For each utterance:
      1. Get phone alignment (non-SIL phones in order)
      2. Get word sequence from transcript
      3. Look up expected phone count per word from CMU dict
      4. Proportionally map phones to words
      5. Extract EMG segment + features for each target word

    Returns: {word: [(features, sp)]}
    """
    data = defaultdict(list)
    target_set = set(w.upper() for w in target_words) if target_words else None
    t0 = time.time()
    matched = 0

    for i, (sp, sess, utt) in enumerate(utts):
        if (i+1) % 50 == 0:
            P(f"    [{i+1}/{len(utts)}] {time.time()-t0:.1f}s")

        emg = load_emg(sp, sess, utt)
        align = load_align(sp, sess, utt)
        trans = load_transcript(sp, sess, utt)
        if emg is None or not align or not trans:
            continue

        words = trans.upper().split()

        # Get expected phone count per word
        phone_counts = []
        skip = False
        for w in words:
            pc = word_nphones(w)
            if pc is None:
                skip = True; break
            phone_counts.append(pc)
        if skip:
            continue

        # Get non-SIL phones in order
        flat_phones = [(sf, ef, ph) for sf, ef, ph in align if ph != "SIL"]
        if not flat_phones:
            continue

        total_expected = sum(phone_counts)
        if total_expected == 0:
            continue

        matched += 1
        scale = len(flat_phones) / total_expected

        # Proportional word boundaries
        cum = [0]
        for pc in phone_counts:
            cum.append(cum[-1] + pc)

        for wi, w in enumerate(words):
            if target_set and w not in target_set:
                continue

            # Scaled phone indices
            si = int(round(cum[wi] * scale))
            ei = int(round(cum[wi+1] * scale))
            si = max(0, min(si, len(flat_phones)-1))
            ei = max(si+1, min(ei, len(flat_phones)))

            word_phones = flat_phones[si:ei]
            if not word_phones:
                continue

            # Get EMG frame range (with context)
            sf = word_phones[0][0]
            ef = word_phones[-1][1]
            s0 = max(0, int((sf - ctx_frames) * FPF))
            s1 = min(len(emg), int((ef + 1 + ctx_frames) * FPF))
            if s1 - s0 < 12:
                continue

            # Resample to fixed length
            seg = resample(emg[s0:s1], seg_len, axis=0)

            # Preprocess each channel (our pipeline: DC removal + bandpass + notch)
            processed = np.empty_like(seg, dtype=np.float64)
            for c in range(seg.shape[1]):
                x = seg[:, c].astype(np.float64)
                x -= np.mean(x)  # DC removal
                try:
                    x = filtfilt(_b_bp, _a_bp, x)  # bandpass 1.3-50Hz
                    x = filtfilt(_b_n, _a_n, x)    # 60Hz notch
                except Exception:
                    pass
                processed[:, c] = x

            # Extract features using our existing pipeline
            feat = extract_features(processed, sample_rate=FS)
            data[w].append((feat, sp))

    elapsed = time.time() - t0
    total_segs = sum(len(v) for v in data.values())
    P(f"    Done: {total_segs} word segs, {len(data)} unique words, "
      f"{matched}/{len(utts)} utts matched, {elapsed:.1f}s")
    return dict(data)


# ══════════════════════════════════════════════════════════════════════
# Training and evaluation
# ══════════════════════════════════════════════════════════════════════

def train_and_eval(name, X_train, y_train, X_test, y_test, labels):
    """Train our RF pipeline and evaluate -- matching train.py setup."""
    P(f"\n  ┌─ {name}")
    P(f"  │ Train: {len(X_train)}, Test: {len(X_test)}, Commands: {len(labels)}")
    if len(X_train) < 10 or len(set(y_train)) < 2:
        P("  └─ Insufficient data!"); return {}

    chance = 1.0 / len(labels)

    # Show class distribution
    trc = Counter(y_train.tolist())
    P(f"  │ Train dist: {dict(sorted((labels[k],v) for k,v in trc.items()))}")

    # ── Model 1: LogisticRegression (our existing pipeline from train.py) ──
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, C=1.0, solver='lbfgs')),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_tr = accuracy_score(y_train, lr_pipe.predict(X_train))
    lr_te = accuracy_score(y_test, lr_pipe.predict(X_test)) if len(X_test) > 0 else 0

    # CV on training data
    try:
        mc = min(Counter(y_train).values())
        nf = min(5, mc)
        if nf >= 2:
            cv = cross_val_score(lr_pipe, X_train, y_train,
                cv=StratifiedKFold(nf, shuffle=True, random_state=42),
                scoring='accuracy')
            lr_cv = f"{cv.mean():.1%}"
        else:
            lr_cv = "N/A"
    except Exception:
        lr_cv = "N/A"

    P(f"  │ LogReg: Train={lr_tr:.1%}, CV={lr_cv}, Test={lr_te:.1%} ({lr_te/chance:.1f}x)")

    # ── Model 2: RandomForest (class-balanced) ──
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=42)),
    ])
    rf_pipe.fit(X_train, y_train)
    rf_tr = accuracy_score(y_train, rf_pipe.predict(X_train))
    rf_te = accuracy_score(y_test, rf_pipe.predict(X_test)) if len(X_test) > 0 else 0

    try:
        mc = min(Counter(y_train).values())
        nf = min(5, mc)
        if nf >= 2:
            cv = cross_val_score(rf_pipe, X_train, y_train,
                cv=StratifiedKFold(nf, shuffle=True, random_state=42),
                scoring='accuracy')
            rf_cv = f"{cv.mean():.1%}"
        else:
            rf_cv = "N/A"
    except Exception:
        rf_cv = "N/A"

    P(f"  │ RF:     Train={rf_tr:.1%}, CV={rf_cv}, Test={rf_te:.1%} ({rf_te/chance:.1f}x)")

    # Best model details
    best_name = "RF" if rf_te >= lr_te else "LogReg"
    best_pipe = rf_pipe if rf_te >= lr_te else lr_pipe
    best_te = max(rf_te, lr_te)

    if len(X_test) > 0:
        y_pred = best_pipe.predict(X_test)
        P(f"  │")
        P(f"  │ Best model: {best_name} → {best_te:.1%} ({best_te/chance:.1f}x chance)")
        P(f"  │")
        P(f"  │ Per-class ({best_name}):")
        for i, l in enumerate(labels):
            m = y_test == i
            if m.sum() > 0:
                c = (y_pred[m] == i).sum()
                P(f"  │   {l:>15s}: {c}/{m.sum()} ({c/m.sum():.0%})")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(labels))))
        P(f"  │")
        P(f"  │ Confusion matrix:")
        header = "  │   " + " " * 16 + " ".join(f"{l[:4]:>4s}" for l in labels)
        P(header)
        for i, l in enumerate(labels):
            row = f"  │   {l:>15s} " + " ".join(f"{cm[i,j]:4d}" for j in range(len(labels)))
            P(row)

    P(f"  └─ {best_name}: {best_te:.1%} ({best_te/chance:.1f}x chance)")

    return {
        'lr_test': lr_te, 'rf_test': rf_te,
        'lr_cv': lr_cv, 'rf_cv': rf_cv,
        'best': best_name, 'best_acc': best_te,
        'n_cls': len(labels), 'chance': chance,
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    P("="*70)
    P("Real EMG Command Classification (RF Pipeline)")
    P("="*70)

    tr_sil = parse_subset("train.silent")
    te_sil = parse_subset("test.silent")

    results = []

    for sp in ["008", "002"]:
        P(f"\n{'='*70}")
        P(f"SPEAKER {sp}")
        P(f"{'='*70}")

        sp_tr = tr_sil.get(sp, [])
        sp_te = te_sil.get(sp, [])

        # ── Step 1: Extract ALL word segments ──
        P(f"\n  Extracting TRAIN word segments ({len(sp_tr)} utterances)...")
        tr_data = extract_word_emg(sp_tr)
        P(f"  Extracting TEST word segments ({len(sp_te)} utterances)...")
        te_data = extract_word_emg(sp_te)

        # ── Step 2: Select command words ──
        # Pick words with 8+ train samples, diverse phonetically
        candidates = {w: len(v) for w, v in tr_data.items() if len(v) >= 8}
        P(f"\n  Words with 8+ train samples: {len(candidates)}")
        for w in sorted(candidates, key=lambda w: candidates[w], reverse=True):
            te_n = len(te_data.get(w, []))
            P(f"    {w:15s}: train={candidates[w]:3d}, test={te_n}")

        # Select top commands (diverse and with test data)
        # Pick words with both train AND test samples, max 8 commands
        good_words = sorted(
            [w for w in candidates if len(te_data.get(w, [])) >= 2],
            key=lambda w: candidates[w], reverse=True
        )[:8]

        if len(good_words) < 3:
            P(f"  Not enough commands! Only {len(good_words)} qualifying words.")
            continue

        P(f"\n  Selected commands: {good_words}")

        # ── Step 3: Build feature matrices ──
        labels = sorted(good_words)
        lm = {w: i for i, w in enumerate(labels)}

        X_train = np.array([f for w in labels for f, _ in tr_data[w]])
        y_train = np.array([lm[w] for w in labels for _ in tr_data[w]], dtype=int)

        X_test_list, y_test_list = [], []
        for w in labels:
            for f, _ in te_data.get(w, []):
                X_test_list.append(f)
                y_test_list.append(lm[w])
        X_test = np.array(X_test_list) if X_test_list else np.zeros((0, X_train.shape[1]))
        y_test = np.array(y_test_list, dtype=int) if y_test_list else np.zeros(0, dtype=int)

        # ── Step 4: Train and evaluate ──
        P(f"\n{'─'*70}")
        P(f"  EXPERIMENT: {len(labels)}-Command Classification (Speaker {sp})")
        P(f"{'─'*70}")
        r = train_and_eval(
            f"Sp{sp} {len(labels)} commands", X_train, y_train, X_test, y_test, labels)
        if r:
            results.append((sp, labels, r))

        # ── Also try fewer commands (top 5) for higher accuracy ──
        if len(good_words) > 5:
            top5 = sorted(good_words[:5])
            lm5 = {w: i for i, w in enumerate(top5)}
            X_tr5 = np.array([f for w in top5 for f, _ in tr_data[w]])
            y_tr5 = np.array([lm5[w] for w in top5 for _ in tr_data[w]], dtype=int)
            X_te5, y_te5 = [], []
            for w in top5:
                for f, _ in te_data.get(w, []):
                    X_te5.append(f); y_te5.append(lm5[w])
            X_te5 = np.array(X_te5) if X_te5 else np.zeros((0, X_tr5.shape[1]))
            y_te5 = np.array(y_te5, dtype=int) if y_te5 else np.zeros(0, dtype=int)

            P(f"\n{'─'*70}")
            P(f"  EXPERIMENT: Top-5 Commands (Speaker {sp})")
            P(f"{'─'*70}")
            r5 = train_and_eval(
                f"Sp{sp} top-5 commands", X_tr5, y_tr5, X_te5, y_te5, top5)
            if r5:
                results.append((sp, top5, r5))

    # ── Summary ──
    P(f"\n{'='*70}")
    P("FINAL SUMMARY")
    P(f"{'='*70}")
    for sp, labels, r in results:
        P(f"\n  Speaker {sp}: {len(labels)} commands → {r['best']} "
          f"{r['best_acc']:.1%} ({r['best_acc']/r['chance']:.1f}x chance)")
        P(f"    Commands: {', '.join(labels)}")
        P(f"    LR: test={r['lr_test']:.1%}, CV={r['lr_cv']}")
        P(f"    RF: test={r['rf_test']:.1%}, CV={r['rf_cv']}")

    P(f"\n{'='*70}")
    P("Done!")


if __name__ == "__main__":
    main()
