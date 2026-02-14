#!/usr/bin/env python3
"""LLM-assisted sentence decoding from noisy phone EMG predictions.

Approach v2 — Multi-level predictions:
  1. Train TWO classifiers:
     a) Manner classifier (6 classes, ~35% accuracy — our most reliable signal)
     b) Phone classifier (all phones, ~6% top-1 but ~20% top-5)
  2. For each test utterance, extract phone segments sequentially
  3. For each segment, predict BOTH manner class AND top-5 phones
  4. Group by word boundaries (SIL markers) and format as structured input
  5. GPT-4.1-nano uses manner patterns + phone candidates to decode words

The key insight: manner classes (vowel/fricative/stop/nasal/etc.) are much
more reliably detected from EMG than individual phones. The pattern
"fricative-stop-vowel-nasal" strongly constrains which English words are possible.
"""

import os, sys, glob, time, warnings
import numpy as np
from collections import Counter, defaultdict
from scipy.signal import resample, butter, filtfilt, iirnotch

from openai import OpenAI

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emg_core.dsp.features import extract_features

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

P = lambda *a, **kw: print(*a, **kw, flush=True)

# ── Constants ──
CORPUS = os.path.expanduser(
    "~/.cache/kagglehub/datasets/xabierdezuazo/emguka-trial-corpus/versions/1/EMG-UKA-Trial-Corpus"
)
FS = 600; FPF = 6.0; NCH = 6; TOTAL_CH = 7
SEG_LEN = 60; CTX_FRAMES = 3; TOP_K = 5

_nyq = FS / 2.0
_b_bp, _a_bp = butter(4, [1.3/_nyq, 50.0/_nyq], btype='band')
_b_n, _a_n = iirnotch(60.0, 30.0, FS)

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
LLM_MODEL = "gpt-4.1-nano"

# Manner of articulation classes (from our earlier experiments)
MANNER = {
    "vowel":     ["IY","IH","EH","AE","AX","AH","UW","UH","AO","AA",
                  "EY","AY","OY","AW","OW","IX","ER","AXR"],
    "nasal":     ["M","N","NG","XN","XM"],
    "fricative": ["S","Z","SH","ZH","F","V","TH","DH","HH"],
    "stop":      ["P","B","T","D","K","G"],
    "approx":    ["L","R","W","Y","XL"],
    "affricate": ["CH","JH"],
}

PHONE_TO_MANNER = {}
for manner, phones in MANNER.items():
    for ph in phones:
        PHONE_TO_MANNER[ph] = manner


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


def extract_feat(emg, sf, ef, ctx=CTX_FRAMES):
    """Extract features for a phone segment with context."""
    s0 = max(0, int((sf - ctx) * FPF))
    s1 = min(len(emg), int((ef + 1 + ctx) * FPF))
    if s1 - s0 < 6: return None
    seg = resample(emg[s0:s1], SEG_LEN, axis=0)
    out = np.empty_like(seg, dtype=np.float64)
    for c in range(seg.shape[1]):
        x = seg[:, c].astype(np.float64); x -= np.mean(x)
        try: x = filtfilt(_b_bp, _a_bp, x); x = filtfilt(_b_n, _a_n, x)
        except: pass
        out[:, c] = x
    return extract_features(out, sample_rate=FS)


# ══════════════════════════════════════════════════════════════════════
# Classifier training
# ══════════════════════════════════════════════════════════════════════

def extract_training_data(utts):
    """Extract phone features and labels from utterances."""
    phone_feats = defaultdict(list)
    t0 = time.time()
    for i, (sp, sess, utt) in enumerate(utts):
        if (i+1) % 20 == 0: P(f"    [{i+1}/{len(utts)}] {time.time()-t0:.1f}s")
        emg = load_emg(sp, sess, utt)
        align = load_align(sp, sess, utt)
        if emg is None or not align: continue
        for sf, ef, ph in align:
            if ph == "SIL" or ef-sf < 3: continue
            f = extract_feat(emg, sf, ef)
            if f is not None:
                phone_feats[ph].append(f)

    P(f"    {sum(len(v) for v in phone_feats.values())} segments, "
      f"{len(phone_feats)} phones, {time.time()-t0:.1f}s")
    return dict(phone_feats)


def train_manner_classifier(phone_feats, min_per_class=10):
    """Train manner-of-articulation classifier (6 classes)."""
    manner_data = defaultdict(list)
    for ph, feats in phone_feats.items():
        m = PHONE_TO_MANNER.get(ph)
        if m:
            manner_data[m].extend(feats)

    labels = sorted(l for l, f in manner_data.items() if len(f) >= min_per_class)
    lm = {l: i for i, l in enumerate(labels)}

    X = np.array([f for l in labels for f in manner_data[l]])
    y = np.array([lm[l] for l in labels for _ in manner_data[l]], dtype=int)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=42)),
    ])
    pipe.fit(X, y)
    acc = accuracy_score(y, pipe.predict(X))
    P(f"    Manner classifier: {len(labels)} classes, {len(X)} samples, train={acc:.1%}")
    return pipe, labels


def train_phone_classifier(phone_feats, min_per_phone=5):
    """Train phone classifier for top-k predictions."""
    valid = sorted(ph for ph, f in phone_feats.items() if len(f) >= min_per_phone)
    lm = {ph: i for i, ph in enumerate(valid)}

    X = np.array([f for ph in valid for f in phone_feats[ph]])
    y = np.array([lm[ph] for ph in valid for _ in phone_feats[ph]], dtype=int)

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=2,
            class_weight='balanced', n_jobs=-1, random_state=42)),
    ])
    pipe.fit(X, y)
    acc = accuracy_score(y, pipe.predict(X))
    P(f"    Phone classifier: {len(valid)} phones, {len(X)} samples, train={acc:.1%}")
    return pipe, valid


# ══════════════════════════════════════════════════════════════════════
# Utterance prediction
# ══════════════════════════════════════════════════════════════════════

def predict_utterance(sp, sess, utt, manner_model, manner_labels,
                      phone_model, phone_labels):
    """Get structured predictions for all phones in an utterance.

    Returns list of word groups (split by SIL), each containing
    phone-level predictions.
    """
    emg = load_emg(sp, sess, utt)
    align = load_align(sp, sess, utt)
    if emg is None or not align:
        return [], []

    words = []  # list of word groups
    current_word = []
    all_phones = []  # flat list for accuracy tracking

    for sf, ef, ph in align:
        if ph == "SIL":
            if current_word:
                words.append(current_word)
                current_word = []
            continue

        if ef - sf < 3:
            continue

        feat = extract_feat(emg, sf, ef)
        if feat is None:
            continue

        feat_2d = feat.reshape(1, -1)

        # Manner prediction
        m_probs = manner_model.predict_proba(feat_2d)[0]
        m_top = np.argsort(m_probs)[::-1][:3]
        manner_preds = [(manner_labels[i], float(m_probs[i])) for i in m_top]

        # Phone prediction
        p_probs = phone_model.predict_proba(feat_2d)[0]
        p_top = np.argsort(p_probs)[::-1][:TOP_K]
        phone_preds = [(phone_labels[i], float(p_probs[i])) for i in p_top]

        entry = {
            'true_phone': ph,
            'true_manner': PHONE_TO_MANNER.get(ph, '?'),
            'manner_preds': manner_preds,
            'phone_preds': phone_preds,
            'duration_ms': round((ef - sf + 1) * 10),  # at 100 FPS
        }
        current_word.append(entry)
        all_phones.append(entry)

    if current_word:
        words.append(current_word)

    return words, all_phones


# ══════════════════════════════════════════════════════════════════════
# LLM decoding
# ══════════════════════════════════════════════════════════════════════

PHONE_EXAMPLES = {
    "AA": "o in father", "AE": "a in cat", "AH": "u in but",
    "AO": "aw in dog", "AW": "ow in cow", "AX": "a in about",
    "AXR": "er in butter", "AY": "i in my", "B": "b", "CH": "ch",
    "D": "d", "DH": "th in the", "EH": "e in bed", "ER": "er in bird",
    "EY": "ay in say", "F": "f", "G": "g", "HH": "h",
    "IH": "i in bit", "IX": "i in roses", "IY": "ee in see",
    "JH": "j", "K": "k", "L": "l", "M": "m", "N": "n",
    "NG": "ng in sing", "OW": "o in go", "OY": "oy in boy", "P": "p",
    "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th in think",
    "UH": "oo in book", "UW": "oo in food", "V": "v", "W": "w",
    "XL": "l in bottle", "XN": "n in button", "XM": "m in bottom",
    "Y": "y", "Z": "z", "ZH": "zh in measure",
}


def format_word_group(word_phones, word_idx):
    """Format one word's worth of phone predictions."""
    n = len(word_phones)
    # Manner sequence (top-1)
    manners = [e['manner_preds'][0][0] for e in word_phones]
    manner_seq = "-".join(manners)
    # Manner with confidence
    manner_detail = ", ".join(
        f"{e['manner_preds'][0][0]}({e['manner_preds'][0][1]:.0%})"
        for e in word_phones
    )

    # Phone candidates per position
    phone_lines = []
    for j, e in enumerate(word_phones):
        preds = e['phone_preds'][:3]  # top-3 phones
        pred_str = "/".join(f"{ph}({PHONE_EXAMPLES.get(ph,ph)})" for ph, _ in preds)
        phone_lines.append(f"    pos{j+1}: {pred_str}")

    return (f"  Word {word_idx}: {n} phones, pattern={manner_seq}\n"
            f"    manner: {manner_detail}\n"
            + "\n".join(phone_lines))


def decode_with_llm(word_groups, client):
    """Send structured predictions to LLM for decoding."""
    n_words = len(word_groups)
    n_phones_total = sum(len(w) for w in word_groups)

    # Format the structured input
    word_descriptions = []
    for i, wg in enumerate(word_groups):
        word_descriptions.append(format_word_group(wg, i+1))

    structured_input = "\n".join(word_descriptions)

    system_msg = """You are an expert at decoding noisy speech from EMG sensors. You will receive phone predictions organized by detected words.

For each word you get:
- Number of phones and manner-of-articulation pattern (vowel/fricative/stop/nasal/approx/affricate)
- Manner predictions have ~35% accuracy (the MOST reliable signal)
- Individual phone candidates have ~6-20% accuracy (noisy but informative as a group)

Your job: figure out what English word each group represents, then combine into a coherent sentence.

Strategy:
1. Use the MANNER PATTERN first — it constrains word shape (e.g., "stop-vowel-stop" → words like "cat", "dog", "bit")
2. Use phone candidates as hints — even if individually unreliable, common candidates across positions narrow options
3. Use the NUMBER OF PHONES — it tells you approximate word length
4. Make the sentence grammatically correct and semantically plausible
5. These are normal English sentences (news/everyday speech)

Output ONLY the decoded sentence in lowercase. No explanation."""

    user_msg = f"""Decode this EMG reading into an English sentence.
Detected {n_words} words, {n_phones_total} total phones.

{structured_input}

Sentence:"""

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=50,
            temperature=0.2,
        )
        decoded = response.choices[0].message.content.strip().lower()
        decoded = decoded.strip('"\'.,!?')
        # Remove any meta-commentary
        for prefix in ["the sentence is:", "decoded:", "answer:"]:
            if decoded.startswith(prefix):
                decoded = decoded[len(prefix):].strip()
        return decoded
    except Exception as e:
        P(f"    LLM error: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════
# Evaluation metrics
# ══════════════════════════════════════════════════════════════════════

def word_error_rate(reference, hypothesis):
    """WER using Levenshtein on words."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=int)
    for i in range(len(ref)+1): d[i][0] = i
    for j in range(len(hyp)+1): d[0][j] = j
    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
    return d[len(ref)][len(hyp)] / max(len(ref), 1)

def word_match(ref, hyp):
    """Fraction of ref words in hypothesis."""
    r = set(ref.lower().split())
    h = set(hyp.lower().split())
    return len(r & h) / max(len(r), 1)

def content_word_match(ref, hyp):
    """Fraction of content words (not stopwords) matched."""
    stops = {'the','a','an','is','are','was','were','in','on','at','to','of',
             'and','or','but','that','this','with','for','it','its','has','have',
             'had','be','been','by','not','they','he','she','we','you'}
    r = set(ref.lower().split()) - stops
    h = set(hyp.lower().split()) - stops
    return len(r & h) / max(len(r), 1)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    P("=" * 70)
    P("LLM-ASSISTED SENTENCE DECODING (v2: Manner + Phone Lattice)")
    P("=" * 70)
    P()
    P("Pipeline: EMG → RF manner predictor (35% acc, 6 classes)")
    P("        + RF phone predictor (top-5, ~20% coverage)")
    P("        → Structured word groups → GPT-4.1-nano → Sentence")
    P()

    client = OpenAI(api_key=OPENAI_KEY)
    tr_sil = parse_subset("train.silent")
    te_sil = parse_subset("test.silent")

    for sp in ["008"]:
        P(f"\n{'=' * 70}")
        P(f"SPEAKER {sp}")
        P(f"{'=' * 70}")

        sp_tr = tr_sil.get(sp, [])
        sp_te = te_sil.get(sp, [])

        # ── Train classifiers (silent data only) ──
        P(f"\n  Training on {len(sp_tr)} silent utterances...")
        phone_feats = extract_training_data(sp_tr)

        P(f"  Training manner classifier...")
        manner_model, manner_labels = train_manner_classifier(phone_feats)
        P(f"  Training phone classifier...")
        phone_model, phone_labels = train_phone_classifier(phone_feats)

        # ── Decode test utterances ──
        P(f"\n  Decoding {len(sp_te)} test utterances...")
        P(f"{'─' * 70}")

        results = []
        manner_accs = []
        phone_top1_accs = []
        phone_topk_accs = []

        for i, (sp_id, sess, utt) in enumerate(sp_te):
            transcript = load_transcript(sp_id, sess, utt)
            if not transcript:
                continue

            word_groups, all_phones = predict_utterance(
                sp_id, sess, utt, manner_model, manner_labels,
                phone_model, phone_labels)

            if not word_groups:
                continue

            # Accuracy metrics
            m_correct = sum(1 for e in all_phones
                            if e['manner_preds'][0][0] == e['true_manner'])
            m_acc = m_correct / max(len(all_phones), 1)
            manner_accs.append(m_acc)

            p1_correct = sum(1 for e in all_phones
                             if e['phone_preds'][0][0] == e['true_phone'])
            p1_acc = p1_correct / max(len(all_phones), 1)
            phone_top1_accs.append(p1_acc)

            pk_correct = sum(1 for e in all_phones
                             if e['true_phone'] in [p for p, _ in e['phone_preds']])
            pk_acc = pk_correct / max(len(all_phones), 1)
            phone_topk_accs.append(pk_acc)

            # Get manner pattern summary
            true_manners = [e['true_manner'] for e in all_phones]
            pred_manners = [e['manner_preds'][0][0] for e in all_phones]

            # Decode with LLM
            llm_decoded = decode_with_llm(word_groups, client)

            wer = word_error_rate(transcript, llm_decoded)
            wm = word_match(transcript, llm_decoded)
            cwm = content_word_match(transcript, llm_decoded)

            results.append({
                'ref': transcript,
                'hyp': llm_decoded,
                'wer': wer,
                'word_match': wm,
                'content_match': cwm,
                'manner_acc': m_acc,
                'phone_top1': p1_acc,
                'phone_top5': pk_acc,
                'n_words_detected': len(word_groups),
                'n_words_true': len(transcript.split()),
                'n_phones': len(all_phones),
            })

            P(f"\n  [{i+1}/{len(sp_te)}] {sp_id}-{sess}-{utt}")
            P(f"  Manner acc: {m_acc:.0%} | Phone: top1={p1_acc:.0%}, top5={pk_acc:.0%}")
            P(f"  Words: detected={len(word_groups)}, true={len(transcript.split())}")
            P(f"  Reference:   \"{transcript}\"")
            P(f"  LLM decoded: \"{llm_decoded}\"")
            P(f"  WER={wer:.0%} | word_match={wm:.0%} | content_match={cwm:.0%}")

        # ══════════════════════════════════════════════════════════════
        # Summary
        # ══════════════════════════════════════════════════════════════

        P(f"\n{'=' * 70}")
        P(f"RESULTS — Speaker {sp}")
        P(f"{'=' * 70}")

        if not results:
            P("  No results!"); continue

        avg_m = np.mean(manner_accs)
        avg_p1 = np.mean(phone_top1_accs)
        avg_pk = np.mean(phone_topk_accs)
        avg_wer = np.mean([r['wer'] for r in results])
        avg_wm = np.mean([r['word_match'] for r in results])
        avg_cwm = np.mean([r['content_match'] for r in results])

        P(f"\n  Classifier accuracy on test data:")
        P(f"    Manner (6 classes):   {avg_m:.1%} (chance=16.7%)")
        P(f"    Phone top-1 (41 cls): {avg_p1:.1%} (chance=2.4%)")
        P(f"    Phone top-5:          {avg_pk:.1%}")

        P(f"\n  LLM decoding metrics:")
        P(f"    Average WER:           {avg_wer:.1%}")
        P(f"    Average word match:    {avg_wm:.1%}")
        P(f"    Average content match: {avg_cwm:.1%}")
        P(f"    Utterances:            {len(results)}")

        P(f"\n  {'─' * 66}")
        P(f"  {'Reference':<35s} {'LLM Output':<25s} WER   WM   CM")
        P(f"  {'─' * 66}")
        for r in results:
            ref = r['ref'][:33]
            hyp = r['hyp'][:23]
            P(f"  {ref:<35s} {hyp:<25s} "
              f"{r['wer']:.0%}  {r['word_match']:.0%}  {r['content_match']:.0%}")

        # Best cases
        by_wer = sorted(results, key=lambda r: r['wer'])
        P(f"\n  Top 3 best (by WER):")
        for r in by_wer[:3]:
            P(f"    WER={r['wer']:.0%} | \"{r['ref']}\"")
            P(f"            → \"{r['hyp']}\"")

        P(f"\n  Top 3 best (by content word match):")
        by_cwm = sorted(results, key=lambda r: -r['content_match'])
        for r in by_cwm[:3]:
            P(f"    CM={r['content_match']:.0%} | \"{r['ref']}\"")
            P(f"              → \"{r['hyp']}\"")

        # Analysis
        P(f"\n{'=' * 70}")
        P("ANALYSIS")
        P(f"{'=' * 70}")
        P(f"\n  The manner classifier provides {avg_m:.1%} accuracy (vs 16.7% chance)")
        P(f"  = {avg_m/0.167:.1f}x above chance on articulatory class detection.")
        P(f"")
        P(f"  Phone top-5 coverage ({avg_pk:.1%}) means ~1 in {1/max(avg_pk,0.01):.0f} phones")
        P(f"  has the correct answer in the candidate list.")
        P(f"")
        P(f"  Word match ({avg_wm:.1%}) measures how many reference words")
        P(f"  appear anywhere in the LLM output (order-independent).")
        P(f"")
        P(f"  Content word match ({avg_cwm:.1%}) measures recovery of")
        P(f"  semantically important words (excluding stopwords).")

    P(f"\n{'=' * 70}")
    P("Done!")
    P(f"{'=' * 70}")


if __name__ == "__main__":
    main()
