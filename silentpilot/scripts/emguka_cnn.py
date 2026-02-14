#!/usr/bin/env python3
"""EMG-UKA CNN v4: Per-speaker audible pretrain → silent fine-tune.

The key insight: per-speaker audible data gives ~8000 segments (enough for CNN).
Pretrain conv features on audible, then fine-tune classifier on silent.

Also fixes: remove over-regularization, train longer, proper LR schedule.
"""

import os, sys, glob, time, warnings
import numpy as np
from collections import Counter, defaultdict
from scipy.signal import resample, butter, filtfilt, iirnotch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emg_core.dsp.features import extract_features

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ── Device ──
if torch.backends.mps.is_available():
    DEV = torch.device("mps")
elif torch.cuda.is_available():
    DEV = torch.device("cuda")
else:
    DEV = torch.device("cpu")
P(f"Device: {DEV}")

# ── Constants ──
CORPUS = os.path.expanduser(
    "~/.cache/kagglehub/datasets/xabierdezuazo/emguka-trial-corpus/versions/1/EMG-UKA-Trial-Corpus"
)
FS = 600; FPF = 6.0; NCH = 6; TOTAL_CH = 7
SEG_LEN = 128

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


# ══════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════

class EMG_CNN(nn.Module):
    """1D CNN: larger capacity, less regularization in conv layers."""
    def __init__(self, in_ch=6, n_cls=6):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: learn low-level EMG patterns
            nn.Conv1d(in_ch, 64, 7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2: mid-level features
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 3: high-level features
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, n_cls),
        )

    def forward(self, x):
        return self.head(self.conv(x))

    def get_features(self, x):
        """Extract conv features (for fine-tuning)."""
        return self.conv(x).squeeze(-1)


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

def preprocess_seg(emg, target_len):
    if len(emg) < 6: return None
    s = resample(emg, target_len, axis=0)
    out = np.empty_like(s, dtype=np.float64)
    for c in range(s.shape[1]):
        x = s[:,c].astype(np.float64); x -= np.mean(x)
        try: x = filtfilt(_b_bp, _a_bp, x); x = filtfilt(_b_n, _a_n, x)
        except: pass
        out[:,c] = x
    return out  # (T, C)

def phone_cls(ph, cm):
    for c, ps in cm.items():
        if ph in ps: return c
    return None


def extract_segments(utts, seg_len=SEG_LEN, ctx=5):
    """Returns {phone: [(raw(C,T), feat(129,))]}"""
    data = defaultdict(list)
    t0 = time.time()
    for i,(sp,sess,utt) in enumerate(utts):
        if (i+1)%50==0: P(f"    [{i+1}/{len(utts)}] {time.time()-t0:.1f}s")
        emg = load_emg(sp,sess,utt); align = load_align(sp,sess,utt)
        if emg is None or not align: continue
        for sf,ef,ph in align:
            if ph=="SIL" or ef-sf<3: continue
            s0 = max(0,int((sf-ctx)*FPF))
            s1 = min(len(emg),int((ef+1+ctx)*FPF))
            if s1-s0<6: continue
            seg = preprocess_seg(emg[s0:s1], seg_len)
            if seg is None: continue
            feat = extract_features(seg, sample_rate=FS)
            data[ph].append((seg.T, feat))  # (C,T) and (129,)
    P(f"    Done: {sum(len(v) for v in data.values())} segs, {time.time()-t0:.1f}s")
    return dict(data)


def build_manner(pdata, min_n=10):
    cd = defaultdict(lambda: ([],[]))
    for ph,items in pdata.items():
        c = phone_cls(ph, MANNER)
        if c:
            for r,f in items: cd[c][0].append(r); cd[c][1].append(f)
    valid = sorted(c for c,(r,f) in cd.items() if len(r)>=min_n)
    if not valid: return None,None,None,[]
    lm = {c:i for i,c in enumerate(valid)}
    Xr,Xf,y = [],[],[]
    for c in valid:
        for r,f in zip(cd[c][0],cd[c][1]):
            Xr.append(r); Xf.append(f); y.append(lm[c])
    return np.array(Xr), np.array(Xf), np.array(y,dtype=int), valid

def build_phones(pdata, top_n=10, min_n=10):
    cts = {p:len(v) for p,v in pdata.items() if p!="SIL" and len(v)>=min_n}
    top = sorted(sorted(cts, key=lambda p:cts[p], reverse=True)[:top_n])
    if len(top)<3: return None,None,None,[]
    lm = {p:i for i,p in enumerate(top)}
    Xr,Xf,y = [],[],[]
    for p in top:
        for r,f in pdata[p]:
            Xr.append(r); Xf.append(f); y.append(lm[p])
    return np.array(Xr), np.array(Xf), np.array(y,dtype=int), top

def align_test(te_data, cmap_or_phones, labels, is_manner=True):
    lm = {l:i for i,l in enumerate(labels)}
    Xr,Xf,y = [],[],[]
    for ph,items in te_data.items():
        if is_manner:
            c = phone_cls(ph, MANNER)
        else:
            c = ph if ph in lm else None
        if c and c in lm:
            for r,f in items:
                Xr.append(r); Xf.append(f); y.append(lm[c])
    if not Xr: return np.zeros((0,NCH,SEG_LEN)), np.zeros((0,129)), np.zeros(0,dtype=int)
    return np.array(Xr), np.array(Xf), np.array(y,dtype=int)


# ══════════════════════════════════════════════════════════════════════
# Normalize + tensorize
# ══════════════════════════════════════════════════════════════════════

def normalize_raw(X_tr, X_te):
    """Per-channel z-normalization."""
    X_tr = X_tr.astype(np.float32).copy()
    X_te = X_te.astype(np.float32).copy() if len(X_te)>0 else np.zeros((0,*X_tr.shape[1:]),dtype=np.float32)
    stats = []
    for c in range(X_tr.shape[1]):
        mu = X_tr[:,c,:].mean(); sd = X_tr[:,c,:].std()+1e-8
        stats.append((mu,sd))
        X_tr[:,c,:] = (X_tr[:,c,:]-mu)/sd
        if len(X_te)>0: X_te[:,c,:] = (X_te[:,c,:]-mu)/sd
    return X_tr, X_te, stats


def class_weights(y, n_cls):
    cw = torch.zeros(n_cls)
    for c,n in Counter(y.tolist()).items(): cw[c] = len(y)/(n_cls*n)
    return cw.to(DEV)


# ══════════════════════════════════════════════════════════════════════
# Training functions
# ══════════════════════════════════════════════════════════════════════

def train_cnn(model, X_tr, y_tr, labels, epochs=200, bs=128, lr=5e-4):
    """Train CNN with NO early stopping -- full training on limited data."""
    n_cls = len(labels)
    cw = class_weights(y_tr, n_cls)
    ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(X_tr)>bs)
    model = model.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2)
    criterion = nn.CrossEntropyLoss(weight=cw)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for xb,yb in dl:
            xb = xb.to(DEV).float(); yb = yb.to(DEV).long()
            # Light augmentation: noise only
            xb = xb + torch.randn_like(xb)*0.05
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step(epoch + 1)

        if (epoch+1)%50==0 or epoch==0:
            model.eval()
            with torch.no_grad():
                pr = model(torch.tensor(X_tr).to(DEV).float()).argmax(1).cpu().numpy()
            P(f"    Ep {epoch+1:3d}: train={accuracy_score(y_tr, pr):.1%} "
              f"({time.time()-t0:.0f}s)")

    return model


def eval_model(model, X_te, y_te, labels, name=""):
    """Evaluate model on test set."""
    if len(X_te)==0:
        P(f"    {name}: No test data!"); return 0.0
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_te).to(DEV).float()).argmax(1).cpu().numpy()
    acc = accuracy_score(y_te, pred)
    chance = 1.0/len(labels)
    P(f"    {name}: {acc:.1%} ({acc/chance:.1f}x chance)")
    for i,l in enumerate(labels):
        m = y_te==i
        if m.sum()>0:
            c = (pred[m]==i).sum()
            P(f"      {l:>12s}: {c}/{m.sum()} ({c/m.sum():.0%})")
    return acc


def train_rf(Xf_tr, y_tr, Xf_te, y_te, labels, name=""):
    """RF baseline on features for comparison."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=12,
                                       class_weight='balanced', n_jobs=-1,
                                       random_state=42)),
    ])
    pipe.fit(Xf_tr, y_tr)
    if len(Xf_te)==0: return 0.0
    pred = pipe.predict(Xf_te)
    acc = accuracy_score(y_te, pred)
    chance = 1.0/len(labels)
    P(f"    {name}: {acc:.1%} ({acc/chance:.1f}x)")
    return acc


# ══════════════════════════════════════════════════════════════════════
# Transfer learning: pretrain on audible → fine-tune on silent
# ══════════════════════════════════════════════════════════════════════

def transfer_learn(aud_Xr, aud_y, sil_Xr_tr, sil_y_tr, sil_Xr_te, sil_y_te,
                   labels, name=""):
    """Pretrain on audible, fine-tune head on silent."""
    P(f"\n  Transfer: {name}")
    n_cls = len(labels)

    # Step 1: Normalize (using audible stats, apply to all)
    Xr_all = np.concatenate([aud_Xr, sil_Xr_tr, sil_Xr_te] if len(sil_Xr_te)>0
                            else [aud_Xr, sil_Xr_tr]).astype(np.float32).copy()
    stats = []
    for c in range(NCH):
        mu = aud_Xr[:,c,:].astype(np.float32).mean()
        sd = aud_Xr[:,c,:].astype(np.float32).std()+1e-8
        stats.append((mu,sd))

    def norm(X):
        X = X.astype(np.float32).copy()
        for c,(mu,sd) in enumerate(stats):
            X[:,c,:] = (X[:,c,:]-mu)/sd
        return X

    aud_n = norm(aud_Xr)
    sil_tr_n = norm(sil_Xr_tr)
    sil_te_n = norm(sil_Xr_te) if len(sil_Xr_te)>0 else sil_Xr_te.astype(np.float32)

    # Step 2: Pretrain on audible (full model)
    P(f"  Pretraining on {len(aud_Xr)} audible samples...")
    model = EMG_CNN(in_ch=NCH, n_cls=n_cls)
    model = train_cnn(model, aud_n, aud_y, labels, epochs=100, lr=5e-4)

    # Evaluate before fine-tuning (audible → silent transfer)
    P(f"\n  Before fine-tuning:")
    eval_model(model, sil_te_n, sil_y_te, labels, "Aud→Sil (no FT)")

    # Step 3: Fine-tune on silent data
    if len(sil_Xr_tr) < 20:
        P("  Not enough silent data for fine-tuning!")
        return eval_model(model, sil_te_n, sil_y_te, labels, "Final")

    P(f"\n  Fine-tuning on {len(sil_Xr_tr)} silent samples...")
    # Freeze conv layers, only train head
    for param in model.conv.parameters():
        param.requires_grad = False
    # Reset head
    model.head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, n_cls),
    ).to(DEV)

    cw = class_weights(sil_y_tr, n_cls)
    criterion = nn.CrossEntropyLoss(weight=cw)
    opt = torch.optim.Adam(model.head.parameters(), lr=1e-3, weight_decay=1e-3)

    ds = TensorDataset(torch.tensor(sil_tr_n), torch.tensor(sil_y_tr))
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    for epoch in range(80):
        model.train()
        for xb,yb in dl:
            xb,yb = xb.to(DEV).float(), yb.to(DEV).long()
            xb = xb + torch.randn_like(xb)*0.05
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
        if (epoch+1)%20==0:
            model.eval()
            with torch.no_grad():
                pr = model(torch.tensor(sil_tr_n).to(DEV).float()).argmax(1).cpu().numpy()
            P(f"    FT Ep {epoch+1}: train={accuracy_score(sil_y_tr, pr):.1%}")

    # Step 4: Unfreeze all and fine-tune with tiny LR
    P(f"\n  Full fine-tune (all layers, tiny LR)...")
    for param in model.parameters():
        param.requires_grad = True
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
    for epoch in range(40):
        model.train()
        for xb,yb in dl:
            xb,yb = xb.to(DEV).float(), yb.to(DEV).long()
            xb = xb + torch.randn_like(xb)*0.03
            opt2.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt2.step()

    P(f"\n  After fine-tuning:")
    acc = eval_model(model, sil_te_n, sil_y_te, labels, "Aud→Sil (FT)")
    return acc


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    P("="*70)
    P("EMG-UKA CNN v4: Pretrain Audible → Fine-tune Silent")
    P("="*70)

    tr_silent = parse_subset("train.silent")
    te_silent = parse_subset("test.silent")
    tr_audible = parse_subset("train.audible")

    results = []

    for sp in ["008", "002"]:
        P(f"\n{'='*70}")
        P(f"SPEAKER {sp}")
        P(f"{'='*70}")

        sp_sil_tr = tr_silent.get(sp,[])
        sp_sil_te = te_silent.get(sp,[])
        sp_aud_tr = tr_audible.get(sp,[])

        P(f"  Silent: {len(sp_sil_tr)} train, {len(sp_sil_te)} test")
        P(f"  Audible: {len(sp_aud_tr)} train")

        # Extract all segments
        P("\n  Extracting silent TRAIN...")
        sil_tr = extract_segments(sp_sil_tr)
        P("  Extracting silent TEST...")
        sil_te = extract_segments(sp_sil_te)
        P("  Extracting audible TRAIN...")
        aud_tr = extract_segments(sp_aud_tr)

        for task_name, build_fn, is_manner in [
            ("Manner", lambda d: build_manner(d), True),
            ("Top-10 Phones", lambda d: build_phones(d, 10), False),
        ]:
            P(f"\n{'─'*70}")
            P(f"  TASK: {task_name}")
            P(f"{'─'*70}")

            # Build datasets
            Xr_sil_tr, Xf_sil_tr, y_sil_tr, labs = build_fn(sil_tr)
            if not labs: P("  No labels!"); continue

            if is_manner:
                Xr_sil_te, Xf_sil_te, y_sil_te = align_test(sil_te, MANNER, labs, True)
                Xr_aud_tr, Xf_aud_tr, y_aud_tr, _ = build_manner(aud_tr)
                Xr_aud_te = Xr_sil_te  # not used for aud eval
            else:
                Xr_sil_te, Xf_sil_te, y_sil_te = align_test(sil_te, None, labs, False)
                Xr_aud_tr, Xf_aud_tr, y_aud_tr, _ = build_phones(aud_tr, 10)

            # Normalize raw data
            Xr_sil_tr_n, Xr_sil_te_n, _ = normalize_raw(Xr_sil_tr, Xr_sil_te)

            # ── Baseline: RF on features ──
            P(f"\n  [1] RF on features (baseline):")
            a = train_rf(Xf_sil_tr, y_sil_tr, Xf_sil_te, y_sil_te, labs, "RF-feat")
            results.append((sp, f"{task_name}-RF", a, len(labs)))

            # ── CNN trained on silent only ──
            P(f"\n  [2] CNN on silent only:")
            cnn = EMG_CNN(in_ch=NCH, n_cls=len(labs))
            cnn = train_cnn(cnn, Xr_sil_tr_n, y_sil_tr, labs, epochs=200, lr=5e-4)
            a = eval_model(cnn, Xr_sil_te_n, y_sil_te, labs, "CNN-silent")
            results.append((sp, f"{task_name}-CNN-sil", a, len(labs)))

            # ── Transfer: audible pretrain → silent fine-tune ──
            if Xr_aud_tr is not None and len(Xr_aud_tr) > 50:
                # Align audible labels to same label set
                if is_manner:
                    # Re-build using same labels
                    Xr_a, Xf_a, y_a = align_test(aud_tr, MANNER, labs, True)
                else:
                    Xr_a, Xf_a, y_a = align_test(aud_tr, None, labs, False)

                if len(Xr_a) > 50:
                    a = transfer_learn(Xr_a, y_a, Xr_sil_tr, y_sil_tr,
                                       Xr_sil_te, y_sil_te, labs,
                                       f"Sp{sp} {task_name}")
                    results.append((sp, f"{task_name}-Transfer", a, len(labs)))

    # ══════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════
    P(f"\n{'='*70}")
    P("FINAL SUMMARY")
    P(f"{'='*70}")
    P(f"  {'Spk':<5s} {'Task':<22s} {'Cls':>3s} {'Acc':>6s} {'xCh':>5s}")
    P(f"  {'─'*5} {'─'*22} {'─'*3} {'─'*6} {'─'*5}")
    for sp,task,acc,nc in results:
        ch = 1.0/nc
        P(f"  {sp:<5s} {task:<22s} {nc:3d} {acc:5.1%} {acc/ch:4.1f}x")
    P(f"{'='*70}")


if __name__ == "__main__":
    main()
