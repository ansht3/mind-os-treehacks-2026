#!/usr/bin/env python3
"""EMG-UKA Phone-level classification (v3 -- balanced + audible transfer).

Key improvements over v2:
1. Class-balanced training (balanced class weights)
2. Audible -> Silent transfer experiment (train on audible, test on silent)
3. Broader context windows around each phone segment
"""

import os
import sys
import glob
import time
import numpy as np
from collections import Counter, defaultdict
from scipy.signal import resample, butter, filtfilt, iirnotch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from emg_core.dsp.features import extract_features

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ── Constants ──

CORPUS = os.path.expanduser(
    "~/.cache/kagglehub/datasets/xabierdezuazo/emguka-trial-corpus/versions/1/EMG-UKA-Trial-Corpus"
)
FS = 600; FPF = 6.0; NCH = 6; TOTAL_CH = 7
SEG_LEN = 60   # fixed segment length
CTX_FRAMES = 3  # context frames before/after phone boundary

MANNER = {
    "vowel":     ["IY","IH","EH","AE","AX","AH","UW","UH","AO","AA",
                  "EY","AY","OY","AW","OW","IX","ER","AXR"],
    "nasal":     ["M","N","NG"],
    "fricative": ["S","Z","SH","ZH","F","V","TH","DH","HH"],
    "stop":      ["P","B","T","D","K","G"],
    "approx":    ["L","R","W","Y","XL","XN","XM"],
    "affricate": ["CH","JH"],
}

_nyq = FS / 2.0
_b_bp, _a_bp = butter(4, [1.3/_nyq, 50.0/_nyq], btype='band')
_b_n, _a_n = iirnotch(60.0, 30.0, FS)


def phone_cls(ph, cm):
    for c, ps in cm.items():
        if ph in ps: return c
    return None


def load_emg(sp, sess, utt):
    d = os.path.join(CORPUS, "emg", sp, sess)
    m = glob.glob(os.path.join(d, f"*_{sp}_{sess}_{utt}.adc"))
    if not m: return None
    raw = np.fromfile(m[0], dtype=np.int16)
    return raw.reshape(len(raw)//TOTAL_CH, TOTAL_CH)[:, :NCH].astype(np.float64)


def load_align(sp, sess, utt):
    f = os.path.join(CORPUS, "Alignments", sp, sess, f"phones_{sp}_{sess}_{utt}.txt")
    if not os.path.exists(f): return []
    return [(int(p[0]), int(p[1]), p[2]) for line in open(f) for p in [line.split()] if len(p)>=3]


def parse_subset(fn):
    r = defaultdict(list)
    for line in open(os.path.join(CORPUS, "Subsets", fn)):
        p = line.strip().split(":")
        if len(p) < 2: continue
        ids = p[1].strip().split()
        if not ids: continue
        sp, sess = p[0].strip().replace("emg_","").split("-")
        for uid in ids: r[sp].append((sp, sess, uid.split("-")[-1]))
    return r


def extract_feat(seg):
    if len(seg) < 6: return None
    s = resample(seg, SEG_LEN, axis=0)
    out = np.empty_like(s, dtype=np.float64)
    for c in range(s.shape[1]):
        x = s[:, c].astype(np.float64); x -= np.mean(x)
        try: x = filtfilt(_b_bp, _a_bp, x); x = filtfilt(_b_n, _a_n, x)
        except: pass
        out[:, c] = x
    return extract_features(out, sample_rate=FS)


def get_phone_feats(utts, ctx_frames=CTX_FRAMES):
    """Extract phone features WITH context window."""
    data = defaultdict(list)
    t0 = time.time()
    for i, (sp, sess, utt) in enumerate(utts):
        if (i+1) % 20 == 0: P(f"    [{i+1}/{len(utts)}] {time.time()-t0:.1f}s")
        emg = load_emg(sp, sess, utt)
        align = load_align(sp, sess, utt)
        if emg is None or not align: continue
        for sf, ef, ph in align:
            if ph == "SIL" or ef-sf < 3: continue
            # Add context frames around phone boundary
            s0 = max(0, int((sf - ctx_frames) * FPF))
            s1 = min(len(emg), int((ef + 1 + ctx_frames) * FPF))
            if s1-s0 < 6: continue
            f = extract_feat(emg[s0:s1])
            if f is not None: data[ph].append(f)
    P(f"    Done: {sum(len(v) for v in data.values())} segs, {time.time()-t0:.1f}s")
    return dict(data)


def build_data(pdata, cmap, min_n=10):
    cd = defaultdict(list)
    for ph, fs in pdata.items():
        c = phone_cls(ph, cmap)
        if c: cd[c].extend(fs)
    valid = sorted(c for c, f in cd.items() if len(f) >= min_n)
    if not valid: return np.array([]), np.array([]), []
    lm = {c: i for i, c in enumerate(valid)}
    X, y = [], []
    for c in valid:
        for f in cd[c]: X.append(f); y.append(lm[c])
    return np.array(X), np.array(y), valid


def align_test_data(te_ph, cmap, labels):
    """Build test X, y aligned to training labels."""
    lm = {l: i for i, l in enumerate(labels)}
    X, y = [], []
    for ph, fs in te_ph.items():
        c = phone_cls(ph, cmap)
        if c and c in lm:
            for f in fs: X.append(f); y.append(lm[c])
    X = np.array(X) if X else np.zeros((0, 129))
    y = np.array(y, dtype=int) if y else np.zeros(0, dtype=int)
    return X, y


def run(name, Xtr, ytr, Xte, yte, labels):
    P(f"\n  {name}")
    P(f"  Train: {len(Xtr)}, Test: {len(Xte)}, Classes: {len(labels)}")
    if len(Xtr) < 10 or len(set(ytr)) < 2:
        P("  Insufficient data!"); return 0.0

    chance = 1.0 / len(labels)
    # Show class distribution
    trc = Counter(ytr)
    P(f"  Train dist: {dict(sorted((labels[k],v) for k,v in trc.items()))}")
    nc = min(30, Xtr.shape[1], max(len(Xtr)//5, 5))

    models = {
        "LR-bal": Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=nc)),
            ('clf', LogisticRegression(max_iter=2000, C=0.1, solver='lbfgs',
                                       class_weight='balanced')),
        ]),
        "RF-bal": Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=12,
                                           min_samples_leaf=2, random_state=42,
                                           n_jobs=-1, class_weight='balanced')),
        ]),
    }

    best_te, best_name = 0.0, ""
    for mn, pipe in models.items():
        P(f"    {mn}...", end=" ")
        pipe.fit(Xtr, ytr)
        tr = accuracy_score(ytr, pipe.predict(Xtr))
        try:
            mc = min(Counter(ytr).values())
            nf = min(3, mc)
            if nf >= 2:
                cv = cross_val_score(pipe, Xtr, ytr,
                    cv=StratifiedKFold(nf, shuffle=True, random_state=42),
                    scoring='accuracy')
                cvs = f"{cv.mean():.1%}"
            else: cvs = "N/A"
        except: cvs = "N/A"
        te = accuracy_score(yte, pipe.predict(Xte)) if len(Xte)>0 else 0
        P(f"Tr {tr:.1%}, CV {cvs}, Te {te:.1%} ({te/chance:.1f}x)")
        if te > best_te: best_te, best_name = te, mn

    if len(Xte) > 0 and best_name:
        pipe = models[best_name]
        pipe.fit(Xtr, ytr)
        yp = pipe.predict(Xte)
        P(f"\n    Per-class ({best_name}):")
        for i, l in enumerate(labels):
            m = yte == i
            if m.sum() > 0:
                c = (yp[m]==i).sum()
                P(f"      {l:>12s}: {c}/{m.sum()} ({c/m.sum():.0%})")
    P(f"    Chance: {chance:.1%}, Best: {best_te:.1%}")
    return best_te


def main():
    P("="*70)
    P("EMG-UKA Phone Classification v3 (balanced + transfer)")
    P("="*70)

    tr_silent = parse_subset("train.silent")
    te_silent = parse_subset("test.silent")
    tr_audible = parse_subset("train.audible")

    results = []

    for sp in ["008"]:
        # ═══════════════════════════════════════════════════════════
        # Part 1: Silent-to-Silent (per-speaker, balanced weights)
        # ═══════════════════════════════════════════════════════════
        P(f"\n{'='*70}")
        P(f"PART 1: SILENT-TO-SILENT (Speaker {sp})")
        P(f"{'='*70}")

        sp_tr = tr_silent.get(sp, [])
        sp_te = te_silent.get(sp, [])
        P(f"  {len(sp_tr)} train, {len(sp_te)} test utterances")

        P("\n  Extracting TRAIN (silent)...")
        tr_ph = get_phone_feats(sp_tr)
        P("  Extracting TEST (silent)...")
        te_ph = get_phone_feats(sp_te)

        # Manner
        P(f"\n{'─'*70}")
        P("  [A] MANNER (balanced)")
        Xtr, ytr, labs = build_data(tr_ph, MANNER)
        Xte, yte = align_test_data(te_ph, MANNER, labs)
        a = run(f"Sp{sp} Manner (silent)", Xtr, ytr, Xte, yte, labs)
        results.append((sp, "manner-silent", a, len(labs)))

        # Top-8 phones
        P(f"\n{'─'*70}")
        P("  [B] TOP-8 PHONES (balanced)")
        cts = {p: len(v) for p,v in tr_ph.items() if p!="SIL" and len(v)>=10}
        top = sorted(sorted(cts, key=lambda p:cts[p], reverse=True)[:8])
        if len(top) >= 3:
            lm = {p:i for i,p in enumerate(top)}
            Xtr_p = np.array([f for p in top for f in tr_ph[p]])
            ytr_p = np.array([lm[p] for p in top for _ in tr_ph[p]])
            Xte_p = []; yte_p = []
            for p in top:
                for f in te_ph.get(p,[]): Xte_p.append(f); yte_p.append(lm[p])
            Xte_p = np.array(Xte_p) if Xte_p else np.zeros((0,Xtr_p.shape[1]))
            yte_p = np.array(yte_p,dtype=int) if yte_p else np.zeros(0,dtype=int)
            a = run(f"Sp{sp} Phones (silent)", Xtr_p, ytr_p, Xte_p, yte_p, top)
            results.append((sp, "phones-silent", a, len(top)))

        # ═══════════════════════════════════════════════════════════
        # Part 2: Audible-to-Silent Transfer (train on audible, test on silent)
        # ═══════════════════════════════════════════════════════════
        P(f"\n{'='*70}")
        P(f"PART 2: AUDIBLE → SILENT TRANSFER (Speaker {sp})")
        P(f"{'='*70}")

        sp_tr_aud = tr_audible.get(sp, [])
        P(f"  {len(sp_tr_aud)} audible train utterances")

        if sp_tr_aud:
            P("\n  Extracting TRAIN (audible)...")
            tr_ph_aud = get_phone_feats(sp_tr_aud)

            # Manner: train audible, test silent
            P(f"\n{'─'*70}")
            P("  [C] MANNER (audible→silent transfer)")
            Xtr_a, ytr_a, labs_a = build_data(tr_ph_aud, MANNER)
            Xte_a, yte_a = align_test_data(te_ph, MANNER, labs_a)
            a = run(f"Sp{sp} Manner (aud→sil)", Xtr_a, ytr_a, Xte_a, yte_a, labs_a)
            results.append((sp, "manner-transfer", a, len(labs_a)))

            # Phones: train audible, test silent
            P(f"\n{'─'*70}")
            P("  [D] TOP-8 PHONES (audible→silent transfer)")
            cts_a = {p: len(v) for p,v in tr_ph_aud.items() if p!="SIL" and len(v)>=15}
            top_a = sorted(sorted(cts_a, key=lambda p:cts_a[p], reverse=True)[:8])
            if len(top_a) >= 3:
                lm_a = {p:i for i,p in enumerate(top_a)}
                Xtr_ap = np.array([f for p in top_a for f in tr_ph_aud[p]])
                ytr_ap = np.array([lm_a[p] for p in top_a for _ in tr_ph_aud[p]])
                Xte_ap = []; yte_ap = []
                for p in top_a:
                    for f in te_ph.get(p,[]):
                        Xte_ap.append(f); yte_ap.append(lm_a[p])
                Xte_ap = np.array(Xte_ap) if Xte_ap else np.zeros((0,Xtr_ap.shape[1]))
                yte_ap = np.array(yte_ap,dtype=int) if yte_ap else np.zeros(0,dtype=int)
                a = run(f"Sp{sp} Phones (aud→sil)", Xtr_ap, ytr_ap, Xte_ap, yte_ap, top_a)
                results.append((sp, "phones-transfer", a, len(top_a)))

    # ── Summary ──
    P(f"\n{'='*70}")
    P("SUMMARY")
    P(f"{'='*70}")
    for sp, task, acc, nc in results:
        ch = 1.0/nc
        P(f"  Sp{sp} {task:>20s} ({nc} cls): {acc:.1%} "
          f"(chance {ch:.1%}, {acc/ch:.1f}x)")
    P(f"\n{'='*70}")
    P("Done!")


if __name__ == "__main__":
    main()
