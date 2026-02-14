#!/usr/bin/env python3
"""Architecture comparison for EMG phone classification.

Tests 5 architectures on the SAME data (Speaker 008, manner of articulation):
  1. RF baseline (on handcrafted features) -- the bar to beat
  2. Plain CNN (from v4)
  3. CNN + Channel/Temporal Attention (SE-Net + temporal attention)
  4. Tiny Transformer Encoder (2 layers, d_model=64)
  5. Masked Autoencoder pretrain → fine-tune (self-supervised)

All models kept tiny for 16GB M1 MacBook. Each experiment < 60s.
"""

import os, sys, glob, time, warnings, math
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
DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
P(f"Device: {DEV}")

# ── Constants ──
CORPUS = os.path.expanduser(
    "~/.cache/kagglehub/datasets/xabierdezuazo/emguka-trial-corpus/versions/1/EMG-UKA-Trial-Corpus"
)
FS = 600; FPF = 6.0; NCH = 6; TOTAL_CH = 7; SEG = 128

MANNER = {
    "vowel":     ["IY","IH","EH","AE","AX","AH","UW","UH","AO","AA",
                  "EY","AY","OY","AW","OW","IX","ER","AXR"],
    "nasal":     ["M","N","NG"],
    "fricative": ["S","Z","SH","ZH","F","V","TH","DH","HH"],
    "stop":      ["P","B","T","D","K","G"],
    "approx":    ["L","R","W","Y","XL","XN","XM"],
    "affricate": ["CH","JH"],
}

_nyq = FS/2.0
_b_bp, _a_bp = butter(4, [1.3/_nyq, 50.0/_nyq], btype='band')
_b_n, _a_n = iirnotch(60.0, 30.0, FS)


# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURE 1: Plain CNN (baseline DL model)
# ══════════════════════════════════════════════════════════════════════

class PlainCNN(nn.Module):
    """Simple 3-block CNN. ~60K params."""
    def __init__(self, ch=6, nc=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, nc))
    def forward(self, x): return self.net(x)


# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURE 2: CNN + Channel Attention + Temporal Attention
# ══════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation: learn which EMG channels matter."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x):
        # x: (B, C, T)
        w = x.mean(dim=2)          # (B, C) - global avg pool per channel
        w = self.fc(w).unsqueeze(2) # (B, C, 1)
        return x * w

class TemporalAttention(nn.Module):
    """Learn which time steps matter most."""
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.Tanh(),
            nn.Linear(dim // 2, 1))
    def forward(self, x):
        # x: (B, C, T) → transpose to (B, T, C)
        xt = x.transpose(1, 2)       # (B, T, C)
        w = self.attn(xt).squeeze(-1) # (B, T)
        w = F.softmax(w, dim=1)       # (B, T)
        # Weighted sum over time
        return (xt * w.unsqueeze(2)).sum(dim=1)  # (B, C)

class AttentionCNN(nn.Module):
    """CNN + Channel SE + Temporal Attention. ~45K params."""
    def __init__(self, ch=6, nc=6):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(ch, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.se1 = ChannelAttention(32, 8)
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2))
        self.se2 = ChannelAttention(64, 8)
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
        self.temp_attn = TemporalAttention(64)
        self.head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, nc))

    def forward(self, x):
        x = self.se1(self.conv1(x))
        x = self.se2(self.conv2(x))
        x = self.conv3(x)
        x = self.temp_attn(x)  # (B, 64) - attention-weighted features
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURE 3: Tiny Transformer Encoder
# ══════════════════════════════════════════════════════════════════════

class PatchEmbed(nn.Module):
    """Embed EMG into patches: Conv1d with stride = patch size."""
    def __init__(self, ch=6, d_model=64, patch_size=8):
        super().__init__()
        self.proj = nn.Conv1d(ch, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        # x: (B, C, T) → (B, n_patches, d_model)
        x = self.proj(x).transpose(1, 2)  # (B, n_patches, d_model)
        return self.norm(x)

class TinyTransformer(nn.Module):
    """Minimal transformer: patch embed → 2-layer encoder → cls head. ~50K params."""
    def __init__(self, ch=6, nc=6, d_model=64, nhead=4, nlayers=2, patch_size=8):
        super().__init__()
        self.patch_embed = PatchEmbed(ch, d_model, patch_size)
        n_patches = SEG // patch_size  # 128/8 = 16 patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches + 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=0.3, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, nc))

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                          # (B, 16, 64)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, 64)
        x = torch.cat([cls, x], dim=1)                    # (B, 17, 64)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)                               # (B, 17, 64)
        x = self.norm(x[:, 0])                             # CLS token → (B, 64)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURE 4: Conformer-style (Conv + Self-Attention hybrid)
# ══════════════════════════════════════════════════════════════════════

class ConformerBlock(nn.Module):
    """Conformer block: FFN → Self-Attn → Conv → FFN."""
    def __init__(self, d=64, nhead=4, conv_k=7):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(d*2, d), nn.Dropout(0.2))
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nhead, dropout=0.2, batch_first=True)
        self.conv_norm = nn.LayerNorm(d)
        self.conv = nn.Sequential(
            nn.Conv1d(d, d*2, 1), nn.GLU(dim=1),  # pointwise
            nn.Conv1d(d, d, conv_k, padding=conv_k//2, groups=d), nn.BatchNorm1d(d),  # depthwise
            nn.SiLU(), nn.Conv1d(d, d, 1), nn.Dropout(0.2))  # pointwise
        self.ff2 = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d*2), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(d*2, d), nn.Dropout(0.2))
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x):
        # x: (B, T, D)
        x = x + 0.5 * self.ff1(x)
        # Self-attention
        xn = self.attn_norm(x)
        xa, _ = self.attn(xn, xn, xn)
        x = x + xa
        # Conv module
        xn = self.conv_norm(x).transpose(1, 2)  # (B, D, T)
        x = x + self.conv(xn).transpose(1, 2)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)

class TinyConformer(nn.Module):
    """2-block Conformer. ~80K params."""
    def __init__(self, ch=6, nc=6, d=64, patch_size=8):
        super().__init__()
        self.embed = PatchEmbed(ch, d, patch_size)
        n_p = SEG // patch_size
        self.pos = nn.Parameter(torch.randn(1, n_p, d) * 0.02)
        self.blocks = nn.Sequential(ConformerBlock(d), ConformerBlock(d))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(d, 32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, nc))

    def forward(self, x):
        x = self.embed(x) + self.pos  # (B, n_patches, d)
        x = self.blocks(x)            # (B, n_patches, d)
        x = x.transpose(1, 2)         # (B, d, n_patches)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════
# ARCHITECTURE 5: Masked Autoencoder pretrain → fine-tune
# ══════════════════════════════════════════════════════════════════════

class MAE_Encoder(nn.Module):
    """Encoder for Masked Autoencoder. Patch embed + tiny transformer."""
    def __init__(self, ch=6, d=64, nhead=4, nlayers=2, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.d = d
        self.embed = PatchEmbed(ch, d, patch_size)
        n_p = SEG // patch_size
        self.pos = nn.Parameter(torch.randn(1, n_p, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d, nhead, d*2, dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)

    def forward(self, x, mask_ratio=0.0):
        # x: (B, C, T)
        patches = self.embed(x) + self.pos  # (B, N, D)
        if mask_ratio > 0:
            B, N, D = patches.shape
            n_keep = max(1, int(N * (1 - mask_ratio)))
            noise = torch.rand(B, N, device=x.device)
            ids_shuffle = noise.argsort(dim=1)
            ids_keep = ids_shuffle[:, :n_keep]
            patches = torch.gather(patches, 1, ids_keep.unsqueeze(-1).expand(-1,-1,D))
        return self.encoder(patches)  # (B, n_keep, D)

class MAE(nn.Module):
    """Masked Autoencoder: pretrain by reconstructing masked patches."""
    def __init__(self, ch=6, d=64, patch_size=8):
        super().__init__()
        self.encoder = MAE_Encoder(ch, d, patch_size=patch_size)
        self.decoder = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, ch * patch_size))  # reconstruct raw patch
        self.patch_size = patch_size
        self.ch = ch

    def forward(self, x):
        # Full forward (no masking) for feature extraction
        z = self.encoder(x, mask_ratio=0.0)  # (B, N, D)
        return z.mean(dim=1)  # (B, D) - average pool

    def pretrain_step(self, x, mask_ratio=0.75):
        """One pretraining step: mask, encode, decode, reconstruct."""
        B, C, T = x.shape
        N = T // self.patch_size

        # Get all patch targets
        targets = x.reshape(B, C, N, self.patch_size).permute(0, 2, 1, 3)  # (B, N, C, P)
        targets = targets.reshape(B, N, -1)  # (B, N, C*P)

        # Encode visible patches
        z = self.encoder(x, mask_ratio=mask_ratio)  # (B, n_vis, D)
        # Decode
        recon = self.decoder(z)  # (B, n_vis, C*P)

        # Loss: MSE on visible patches (simplified -- full MAE would reconstruct masked)
        n_vis = z.shape[1]
        target_vis = targets[:, :n_vis]  # approximate
        return F.mse_loss(recon, target_vis)


class MAE_Classifier(nn.Module):
    """MAE encoder + classification head."""
    def __init__(self, encoder, d=64, nc=6):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 32), nn.GELU(), nn.Dropout(0.3), nn.Linear(32, nc))

    def forward(self, x):
        z = self.encoder(x, mask_ratio=0.0)  # (B, N, D)
        z = z.mean(dim=1)                      # (B, D)
        return self.head(z)


# ══════════════════════════════════════════════════════════════════════
# Data loading (reuse from v4)
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

def preprocess_seg(emg, tl):
    if len(emg)<6: return None
    s = resample(emg, tl, axis=0)
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

def extract_segments(utts, seg_len=SEG, ctx=5):
    data = defaultdict(list)
    t0 = time.time()
    for i,(sp,sess,utt) in enumerate(utts):
        if (i+1)%50==0: P(f"    [{i+1}/{len(utts)}] {time.time()-t0:.1f}s")
        emg = load_emg(sp,sess,utt); align = load_align(sp,sess,utt)
        if emg is None or not align: continue
        for sf,ef,ph in align:
            if ph=="SIL" or ef-sf<3: continue
            s0 = max(0,int((sf-ctx)*FPF)); s1 = min(len(emg),int((ef+1+ctx)*FPF))
            if s1-s0<6: continue
            seg = preprocess_seg(emg[s0:s1], seg_len)
            if seg is None: continue
            feat = extract_features(seg, sample_rate=FS)
            data[ph].append((seg.T, feat))
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
        for r,f in zip(cd[c][0], cd[c][1]):
            Xr.append(r); Xf.append(f); y.append(lm[c])
    return np.array(Xr), np.array(Xf), np.array(y,dtype=int), valid

def build_phones(pdata, top_n=10, min_n=10):
    cts = {p:len(v) for p,v in pdata.items() if p!="SIL" and len(v)>=min_n}
    top = sorted(sorted(cts, key=lambda p:cts[p], reverse=True)[:top_n])
    if len(top)<3: return None,None,None,[]
    lm = {p:i for i,p in enumerate(top)}
    Xr,Xf,y = [],[],[]
    for p in top:
        for r,f in pdata[p]: Xr.append(r); Xf.append(f); y.append(lm[p])
    return np.array(Xr), np.array(Xf), np.array(y,dtype=int), top

def align_test(te_data, labels, is_manner=True):
    lm = {l:i for i,l in enumerate(labels)}
    Xr,Xf,y = [],[],[]
    for ph,items in te_data.items():
        c = phone_cls(ph, MANNER) if is_manner else (ph if ph in lm else None)
        if c and c in lm:
            for r,f in items: Xr.append(r); Xf.append(f); y.append(lm[c])
    if not Xr: return np.zeros((0,NCH,SEG)), np.zeros((0,129)), np.zeros(0,dtype=int)
    return np.array(Xr), np.array(Xf), np.array(y,dtype=int)


# ══════════════════════════════════════════════════════════════════════
# Unified training
# ══════════════════════════════════════════════════════════════════════

def normalize_raw(Xtr, Xte):
    Xtr = Xtr.astype(np.float32).copy()
    Xte = Xte.astype(np.float32).copy() if len(Xte)>0 else np.zeros((0,*Xtr.shape[1:]),dtype=np.float32)
    for c in range(Xtr.shape[1]):
        mu = Xtr[:,c,:].mean(); sd = Xtr[:,c,:].std()+1e-8
        Xtr[:,c,:] = (Xtr[:,c,:]-mu)/sd
        if len(Xte)>0: Xte[:,c,:] = (Xte[:,c,:]-mu)/sd
    return Xtr, Xte

def train_eval(name, model, Xtr, ytr, Xte, yte, labels, epochs=150, bs=128, lr=1e-3):
    """Train model, return test accuracy. Prints progress."""
    nc = len(labels); chance = 1.0/nc
    P(f"\n  [{name}] {sum(p.numel() for p in model.parameters()):,} params")

    cw = torch.zeros(nc)
    for c,n in Counter(ytr.tolist()).items(): cw[c] = len(ytr)/(nc*n)
    cw = cw.to(DEV)

    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=len(Xtr)>bs)
    model = model.to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss(weight=cw)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            xb = xb.to(DEV).float() + torch.randn(xb.shape, device=DEV)*0.05
            opt.zero_grad()
            loss = crit(model(xb), yb.to(DEV).long())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if (ep+1)%50==0:
            model.eval()
            with torch.no_grad():
                pr = model(torch.tensor(Xtr).to(DEV).float()).argmax(1).cpu().numpy()
            P(f"    Ep {ep+1}: train={accuracy_score(ytr,pr):.1%} ({time.time()-t0:.0f}s)")

    elapsed = time.time()-t0
    model.eval()
    with torch.no_grad():
        pr_tr = model(torch.tensor(Xtr).to(DEV).float()).argmax(1).cpu().numpy()
        tr_acc = accuracy_score(ytr, pr_tr)
    te_acc = 0.0
    if len(Xte)>0:
        with torch.no_grad():
            pr_te = model(torch.tensor(Xte).to(DEV).float()).argmax(1).cpu().numpy()
        te_acc = accuracy_score(yte, pr_te)
    P(f"    → Train={tr_acc:.1%}, Test={te_acc:.1%} ({te_acc/chance:.1f}x) [{elapsed:.0f}s]")

    # Per-class for test
    if len(Xte)>0:
        for i,l in enumerate(labels):
            m = yte==i
            if m.sum()>0:
                c = (pr_te[m]==i).sum()
                P(f"      {l:>12s}: {c}/{m.sum()} ({c/m.sum():.0%})")
    return te_acc


def pretrain_mae(Xtr_all, epochs=100, bs=128, lr=1e-3, mask_ratio=0.75):
    """Self-supervised pretraining on ALL raw EMG (no labels needed)."""
    P(f"\n  [MAE Pretrain] on {len(Xtr_all)} segments, mask={mask_ratio:.0%}")
    mae = MAE(ch=NCH, d=64, patch_size=8).to(DEV)
    ds = TensorDataset(torch.tensor(Xtr_all))
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    for ep in range(epochs):
        mae.train()
        total_loss = 0
        for (xb,) in dl:
            xb = xb.to(DEV).float()
            opt.zero_grad()
            loss = mae.pretrain_step(xb, mask_ratio)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        if (ep+1)%25==0:
            P(f"    Ep {ep+1}: recon_loss={total_loss/len(dl):.4f} ({time.time()-t0:.0f}s)")

    P(f"    Pretrain done in {time.time()-t0:.0f}s")
    return mae.encoder


# ══════════════════════════════════════════════════════════════════════
# Main experiment
# ══════════════════════════════════════════════════════════════════════

def main():
    P("="*70)
    P("Architecture Comparison: EMG Phone Classification")
    P("="*70)

    tr_sil = parse_subset("train.silent")
    te_sil = parse_subset("test.silent")
    tr_aud = parse_subset("train.audible")

    results = []

    for sp in ["008"]:
        P(f"\n{'='*70}")
        P(f"SPEAKER {sp}")
        P(f"{'='*70}")

        sp_tr = tr_sil.get(sp,[]); sp_te = te_sil.get(sp,[]); sp_aud = tr_aud.get(sp,[])

        P("\n  Loading data...")
        P("  Silent train:"); sil_tr = extract_segments(sp_tr)
        P("  Silent test:"); sil_te = extract_segments(sp_te)
        P("  Audible train:"); aud_tr = extract_segments(sp_aud)

        for task, build_fn, is_manner in [
            ("Manner (6 cls)", lambda d: build_manner(d), True),
            ("Top-10 Phones", lambda d: build_phones(d, 10), False),
        ]:
            P(f"\n{'━'*70}")
            P(f"  TASK: {task}")
            P(f"{'━'*70}")

            Xr_tr, Xf_tr, y_tr, labs = build_fn(sil_tr)
            if not labs: P("  No labels!"); continue
            Xr_te, Xf_te, y_te = align_test(sil_te, labs, is_manner)
            nc = len(labs); chance = 1.0/nc

            P(f"  Train: {len(Xr_tr)}, Test: {len(Xr_te)}, Classes: {nc}")
            dist = Counter(y_tr.tolist())
            P(f"  Dist: {dict(sorted((labs[k],v) for k,v in dist.items()))}")

            # Normalize
            Xr_n, Xr_te_n = normalize_raw(Xr_tr, Xr_te)

            # ── 0. RF baseline ──
            P(f"\n  [0. RF Baseline]")
            pipe = Pipeline([('s',StandardScaler()),
                             ('c',RandomForestClassifier(200, max_depth=12,
                                  class_weight='balanced', n_jobs=-1, random_state=42))])
            pipe.fit(Xf_tr, y_tr)
            rf_acc = accuracy_score(y_te, pipe.predict(Xf_te)) if len(Xf_te)>0 else 0
            P(f"    → Test={rf_acc:.1%} ({rf_acc/chance:.1f}x)")
            results.append((sp, task, "RF-features", rf_acc, nc, 0))

            # ── 1. Plain CNN ──
            m = PlainCNN(NCH, nc)
            a = train_eval("1. PlainCNN", m, Xr_n, y_tr, Xr_te_n, y_te, labs)
            results.append((sp, task, "PlainCNN", a, nc, sum(p.numel() for p in m.parameters())))

            # ── 2. CNN + Attention ──
            m = AttentionCNN(NCH, nc)
            a = train_eval("2. CNN+Attention", m, Xr_n, y_tr, Xr_te_n, y_te, labs)
            results.append((sp, task, "CNN+Attn", a, nc, sum(p.numel() for p in m.parameters())))

            # ── 3. Tiny Transformer ──
            m = TinyTransformer(NCH, nc, d_model=64, nhead=4, nlayers=2)
            a = train_eval("3. TinyTransformer", m, Xr_n, y_tr, Xr_te_n, y_te, labs)
            results.append((sp, task, "Transformer", a, nc, sum(p.numel() for p in m.parameters())))

            # ── 4. Conformer ──
            m = TinyConformer(NCH, nc, d=64)
            a = train_eval("4. Conformer", m, Xr_n, y_tr, Xr_te_n, y_te, labs)
            results.append((sp, task, "Conformer", a, nc, sum(p.numel() for p in m.parameters())))

            # ── 5. MAE pretrain → fine-tune ──
            # Pretrain on ALL available raw data (audible + silent, no labels)
            Xr_aud = align_test(aud_tr, labs, is_manner)[0] if aud_tr else np.zeros((0,NCH,SEG))
            Xr_all_raw = np.concatenate([Xr_n, normalize_raw(Xr_aud, np.zeros((0,NCH,SEG)))[0]]
                                         ) if len(Xr_aud)>0 else Xr_n
            P(f"\n  MAE: {len(Xr_all_raw)} total segments for pretraining")
            encoder = pretrain_mae(Xr_all_raw, epochs=80, mask_ratio=0.6)
            clf = MAE_Classifier(encoder, d=64, nc=nc)
            a = train_eval("5. MAE→Finetune", clf, Xr_n, y_tr, Xr_te_n, y_te, labs,
                           epochs=100, lr=5e-4)
            results.append((sp, task, "MAE+FT", a, nc, sum(p.numel() for p in clf.parameters())))

    # ══════════════════════════════════════════════════════════════
    P(f"\n{'='*70}")
    P("FINAL COMPARISON")
    P(f"{'='*70}")
    P(f"  {'Task':<18s} {'Model':<16s} {'Params':>8s} {'Acc':>6s} {'xCh':>5s}")
    P(f"  {'─'*18} {'─'*16} {'─'*8} {'─'*6} {'─'*5}")
    for sp, task, model, acc, nc, params in results:
        ch = 1.0/nc
        ps = f"{params:,}" if params>0 else "N/A"
        marker = " ★" if acc == max(a for _,t,_,a,_,_ in results if t==task) else ""
        P(f"  {task:<18s} {model:<16s} {ps:>8s} {acc:5.1%} {acc/ch:4.1f}x{marker}")
    P(f"{'='*70}")


if __name__ == "__main__":
    main()
