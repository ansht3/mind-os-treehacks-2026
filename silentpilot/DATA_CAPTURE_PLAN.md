# SilentPilot Data Capture & Accuracy Expectations

## Master Planning Document for Hackathon Sensor Deployment

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Reference Benchmarks from Published Research](#2-reference-benchmarks)
3. [Hardware Setup & Electrode Placement](#3-hardware-setup)
4. [Data Capture Protocol](#4-data-capture-protocol)
5. [Expected Accuracy: Command Classification](#5-accuracy-commands)
6. [Expected Accuracy: Phone-Level Recognition](#6-accuracy-phones)
7. [Expected Accuracy: LLM-Assisted Free Speech](#7-accuracy-llm)
8. [Mathematical Derivations](#8-math)
9. [Session Planning: The 2-Hour Calibration](#9-session-plan)
10. [Failure Modes & Contingencies](#10-failure-modes)
11. [Quick-Start Checklist](#11-checklist)

---

## 1. The Big Picture

We have three operating modes, each requiring different data and yielding different accuracy:

| Mode | What it does | Data needed | Expected accuracy | Feasibility |
|---|---|---|---|---|
| **Command classification** | Classify isolated silent words into 5–10 commands | 50–75 samples/command | **80–93%** | High — proven |
| **Phone recognition** | Classify individual phoneme segments | 200+ samples/phone | **40–60% top-1** | Medium — novel for us |
| **LLM free speech** | Phone lattice → LLM decodes sentences | Phone model + LLM API | **30–60% word accuracy** | Experimental |

The critical insight: these modes are **layered, not exclusive**. We build the phone model as a superset of the command model, and the LLM decoder sits on top of the phone model. A single 2-hour calibration session feeds all three modes.

---

## 2. Reference Benchmarks from Published Research

### 2.1 Isolated Word/Command Classification

| System | Words | Electrodes | Samples/class | Accuracy | Year |
|---|---|---|---|---|---|
| **AlterEgo (MIT)** | 10 digits | 7 face/jaw | 75 | **92% median** | 2018 |
| Meltzner et al. | 65 words | 11 face/neck | ~50 | **93% (5 sensors)** | 2011 |
| Wearable headphone EMG | 10 commands | 4 textile | ~100 | **96%** | 2025 |
| Silent speech AR interface | 10 words | 4 sEMG | ~50 | **82.5% avg** | 2022 |
| **Our EMG-UKA result** | 5 words | 6 ch | ~15 (noisy) | **62.5%** | 2026 |

**Key observations:**
- With dedicated hardware and 50–75 samples/class, 85–96% is consistently achieved in the literature
- Our 62.5% was on extracted words from continuous speech (the hardest scenario) with only ~15 clean samples per class
- The gap between our result and the literature (~30pp) is almost entirely explained by: noisy segmentation, too few samples, and non-optimized electrode placement

### 2.2 Continuous Speech / Open Vocabulary

| System | Task | Data | WER | Year |
|---|---|---|---|---|
| **Gaddy (Berkeley)** | Open vocab, silent | 20 hours + DNN | **28% (vocalized), 68% (silent)** | 2020 |
| Wand & Schultz | EMG-UKA continuous | 7.3 hours, DNN-HMM | ~60% WER (audible) | 2016 |
| Gaddy improved | Open vocab, silent | 20 hours + transfer | **4% (vocalized), 68% (silent)** | 2021 |

**Key observations:**
- State-of-the-art open-vocabulary silent speech WER is ~68% with 20 hours of data and deep neural networks
- Vocalized (audible) EMG achieves 4% WER — the gap between audible and silent is the fundamental challenge
- These systems use sophisticated encoder-decoder architectures trained on parallel audible+silent data

### 2.3 What This Means For Us

With 2 hours of data (not 20), classical ML (not large DNNs), and without parallel audio targets, our ceiling is lower than Gaddy's. But we have a weapon they didn't: **an LLM as the language model**. GPT-4.1-nano has far stronger language priors than any n-gram or small neural LM from 2020.

---

## 3. Hardware Setup & Electrode Placement

### 3.1 Optimal Electrode Positions

Based on AlterEgo and Meltzner et al., the most informative muscle groups for silent speech are:

```
         ┌──────────────┐
         │   FOREHEAD    │  ← Reference/Ground electrode
         │    (GND)      │
         └──────────────┘
              │
    ┌────────┴────────┐
    │  FACE (front)    │
    │                  │
    │  [1] Buccal R    │  ← Right cheek (buccinator muscle)
    │  [2] Buccal L    │  ← Left cheek
    │  [3] Masseter R  │  ← Right jaw (chewing muscle)
    │                  │
    └────────┬────────┘
             │
    ┌────────┴────────┐
    │  CHIN / NECK     │
    │                  │
    │  [4] Submental   │  ← Below chin (tongue/hyoid)
    │  [5] Laryngeal   │  ← Throat (vocal folds)
    │  [6] Infrahyoid  │  ← Below Adam's apple
    │                  │
    └─────────────────┘
```

**Priority order** (if limited channels):
1. **Submental** (below chin) — highest information for tongue movements
2. **Laryngeal** (throat) — captures vocal fold activity
3. **Masseter** (jaw) — jaw opening/closing
4. **Buccal** (cheek) — lip rounding, cheek tension
5. **Infrahyoid** — secondary laryngeal information
6. **Additional facial positions** — diminishing returns after 5 channels

Research shows **5 channels capture >99% of the information** from an 11-channel setup (Meltzner et al., 2011). With our 6-channel system we should have sufficient coverage.

### 3.2 Electrode Application Protocol

1. **Clean skin** with alcohol wipe at each site
2. **Apply conductive gel** to Ag/AgCl electrodes
3. **Attach firmly** — electrode shift is the #1 accuracy killer
4. **Mark positions** with a skin-safe marker for re-application between sessions
5. **Check impedance** — should be <10 kΩ for clean signal
6. **Verify signal** — have user clench jaw, swallow, stick out tongue; each should produce visible deflection on the corresponding channels

### 3.3 Signal Quality Verification

Before collecting data, verify the setup captures articulatory information:

| Action | Expected channels active | If missing |
|---|---|---|
| Clench jaw | Masseter (ch 3) | Re-seat jaw electrode |
| Stick out tongue | Submental (ch 4) | Re-seat chin electrode |
| Swallow | Laryngeal (ch 5), Infrahyoid (ch 6) | Re-seat throat electrodes |
| Purse lips | Buccal (ch 1, 2) | Re-seat cheek electrodes |
| Say "EEE" silently | Multiple channels | Full check needed |

---

## 4. Data Capture Protocol

### 4.1 Overview: The Three-Phase Session

A comprehensive 2-hour calibration session is divided into three phases, each feeding a different capability:

| Phase | Duration | What | Feeds |
|---|---|---|---|
| **Phase 1: Commands** | 30 min | Isolated command words, PTT | Command classifier |
| **Phase 2: Phone Drill** | 30 min | Isolated words covering all phones | Phone classifier |
| **Phase 3: Free Speech** | 60 min | Read sentences aloud silently | Phone model + LLM decoder |

**Rest breaks:** 5 minutes between phases. This also provides session-independence data, which is critical for generalization.

### 4.2 Phase 1: Command Calibration (30 minutes)

**Goal:** Collect 75 samples per command for 8 commands = 600 samples

**Protocol:**
1. Display command word on screen with 3-second countdown
2. User presses and holds PTT button
3. User silently vocalizes the word (internally speaks it, no lip movement or sound)
4. User releases PTT button
5. System records the segment and shows visual confirmation
6. 1-second pause, then next command appears (randomized order)

**Commands (recommended set):**

| Command | Phones | Why it's good |
|---|---|---|
| OPEN | OW-P-AH-N | Bilabial + nasal, distinctive lip movement |
| STOP | S-T-AA-P | Fricative onset, back vowel |
| YES | Y-EH-S | Palatal + front vowel + fricative |
| NO | N-OW | Nasal + back vowel diphthong |
| NEXT | N-EH-K-S-T | Complex consonant cluster |
| BACK | B-AE-K | Bilabial + velar endpoints |
| SCROLL | S-K-R-OW-L | Long, multi-manner sequence |
| CLICK | K-L-IH-K | Velar stops bracket a vowel |

**Timing:**
- 75 samples × 8 commands = 600 utterances
- ~3 seconds per utterance (including pause)
- Total: ~30 minutes

**Mid-phase checkpoint (after 30 samples/command):**
- Run quick RF training on collected data
- Display per-class accuracy
- If any command <50% accuracy, consider replacing it with an alternative

### 4.3 Phase 2: Phone Drilling (30 minutes)

**Goal:** Collect ~100 samples per phoneme class by reading isolated words

**Protocol:**
- Display a word on screen; user silently vocalizes it with PTT
- Each word targets specific phones
- Words selected to cover all 41 phone classes with balanced distribution

**Word list design** (examples — each word covers 2–5 phones):

| Target phones | Word | Phones |
|---|---|---|
| P, IH, T | PIT | P-IH-T |
| B, AE, T | BAT | B-AE-T |
| K, AE, T | CAT | K-AE-T |
| S, IY | SEE | S-IY |
| DH, AH | THE | DH-AH |
| CH, IY, Z | CHEESE | CH-IY-Z |
| SH, UW | SHOE | SH-UW |
| M, AW, TH | MOUTH | M-AW-TH |
| F, IH, SH | FISH | F-IH-SH |
| R, EH, D | RED | R-EH-D |

**Why isolated words, not sentences:**
- Cleaner segmentation (PTT marks entire word)
- Phone boundaries estimated from CMU dict (proportional mapping)
- More balanced phone distribution than natural sentences

**Timing:**
- ~600 words × 3s = 30 minutes
- Expected yield: ~2400 phone segments, ~60 per phone class

### 4.4 Phase 3: Free Speech / Sentence Reading (60 minutes)

**Goal:** Collect continuous speech data for phone model refinement and LLM decoder training

**Protocol:**
- Display a sentence on screen
- User reads it silently with PTT held for entire sentence
- System records the full-sentence EMG segment
- Forced alignment (using our CMU dict proportional mapping) segments into phones

**Sentence source:** Use a balanced corpus. Options:
- TIMIT sentence list (phonetically balanced)
- Harvard sentences (phonetically balanced, short)
- News headlines (natural, varied vocabulary)
- Custom sentences using our command vocabulary (for domain adaptation)

**Timing:**
- Average sentence: ~8 words, ~5 seconds to vocalize, ~3s gap = 8s cycle
- 60 minutes ÷ 8s = ~450 sentences
- ~450 × 8 words × 4.5 phones/word = **~16,200 phone segments**

**Combined with Phase 2:** Total phone segments ≈ 2,400 + 16,200 = **~18,600**
Per phone class (41 phones): **~450 samples/phone** (assuming balanced distribution)

This is a realistic estimate. In practice, common phones (T, N, S, DH) will have 800+ samples while rare phones (ZH, OY) may have <100.

---

## 5. Expected Accuracy: Command Classification

### 5.1 Estimates by Data Amount

Based on our experiments, AlterEgo's results, and the broader sEMG literature:

| Samples/cmd | 5 commands | 8 commands | 10 commands |
|---|---|---|---|
| 10 | 50–60% | 35–50% | 30–45% |
| 25 | 65–75% | 55–65% | 50–60% |
| **50** | **80–88%** | **72–82%** | **68–78%** |
| **75** | **88–93%** | **82–90%** | **80–88%** |
| 100 | 90–95% | 88–93% | 85–92% |
| 150 | 92–96% | 90–95% | 88–93% |

### 5.2 Justification

**Lower bound** comes from our EMG-UKA results, adjusted for clean PTT segmentation (+15pp) and phonetically diverse commands (+5pp):
- EMG-UKA 5 commands, ~15 samples: 62.5% → Adjusted: ~82% with 50 samples

**Upper bound** comes from the published literature:
- AlterEgo: 92% on 10 classes with 75 samples
- Meltzner: 93% on 65 words with 5 electrodes
- Wearable headphone: 96% on 10 commands

**Our best estimate for hackathon day (8 commands, 75 samples):** **85 ± 5%**

### 5.3 Confidence Levels

| Confidence | Scenario | Accuracy |
|---|---|---|
| **Very likely (>80%)** | 5 commands, 75 samples, good electrodes | 85–93% |
| **Likely (>60%)** | 8 commands, 50 samples, decent electrodes | 75–85% |
| **Possible (~40%)** | 8 commands, 50 samples, mediocre electrodes | 60–75% |
| **Unlikely (<20%)** | Any scenario where electrode signal is poor | <50% |

The single biggest risk factor is **electrode signal quality**, not software or data quantity.

---

## 6. Expected Accuracy: Phone-Level Recognition

### 6.1 The Phone Accuracy Problem

Phone recognition is fundamentally harder than command classification because:
1. **More classes** (41 vs. 8) — chance drops from 12.5% to 2.4%
2. **Shorter segments** (~100ms vs. ~500ms) — less EMG signal per segment
3. **Confusable pairs** — many phones have similar articulatory patterns (P/B, T/D, S/Z)
4. **Coarticulation** — phone EMG varies depending on surrounding phones

### 6.2 Estimates by Data Amount

| Samples/phone | Top-1 (41 cls) | Top-3 | Top-5 | Manner (6 cls) |
|---|---|---|---|---|
| 50 | 12–18% | 25–35% | 35–45% | 35–45% |
| 200 | 20–30% | 40–50% | 50–60% | 50–60% |
| **450 (our 2hr plan)** | **30–45%** | **50–65%** | **60–75%** | **60–75%** |
| 1000 | 40–55% | 60–75% | 70–85% | 70–80% |
| 2000 | 45–60% | 65–80% | 75–90% | 75–85% |

### 6.3 Justification

**Our EMG-UKA baseline:** 5.7% top-1 on 41 phones (test set, cross-session, ~50 samples/phone from continuous speech). This is the absolute worst case: noisy extraction, cross-session, minimal data.

**Expected improvements with our setup:**
- **Same-session training/testing**: eliminates electrode shift drift → +10–15pp
- **PTT segmentation** (clean boundaries): reduces segment noise → +5–10pp
- **450 samples/phone** (vs. 50): more training data → +10–15pp
- **Combined effect**: 5.7% → estimated **30–45% top-1**

**Cross-reference with literature:**
- AlterEgo achieved 92% on 10 words (each word is a phone sequence) with 75 samples — this implies strong phone-level discriminability exists in the EMG signal
- Gaddy achieved 68% sentence WER with DNNs on 20 hours of data, which requires ~50–60% phone accuracy as a minimum
- Wand 2016 showed 32% relative improvement with DNNs over GMMs, suggesting our RF-based system has room to grow

### 6.4 Manner-of-Articulation as Intermediate Target

If individual phone accuracy is too low for useful LLM decoding, **manner classes** (6 categories) provide a more reliable signal:

| Manner class | Phones included | EMG signature |
|---|---|---|
| **Vowel** | IY, IH, EH, AE, AH, UW, etc. | Sustained laryngeal + oral cavity activity |
| **Nasal** | M, N, NG | Nasal resonance, velum lowering |
| **Fricative** | S, Z, F, V, TH, DH, SH | High-frequency turbulence activity |
| **Stop** | P, B, T, D, K, G | Brief burst after silence |
| **Approximant** | L, R, W, Y | Smooth articulatory movement |
| **Affricate** | CH, JH | Stop + fricative combination |

With 450 samples/phone, manner classification should reach **60–75% accuracy**, providing a strong structural signal for the LLM even when individual phones are uncertain.

---

## 7. Expected Accuracy: LLM-Assisted Free Speech

### 7.1 Architecture

```
EMG segments (sequential) → Phone classifier (top-5 per segment)
                          → Manner classifier (top-3 per segment)
                          → Word boundary detection (SIL/timing)
                          → Structured lattice
                          → GPT-4.1-nano (or better)
                          → Decoded English sentence
```

### 7.2 The Math: Phone Accuracy → Word Recovery

An average English word has **4.5 phonemes**. The probability that a word's phone sequence is fully covered in the top-k predictions determines decodability:

| Phone top-5 acc | P(all phones in top-5) | With dictionary | With LLM context | Estimated word acc |
|---|---|---|---|---|
| 20% (our EMG-UKA) | 0.03% | ~1% | ~2% | **~0%** |
| 40% | 1.6% | ~5% | ~10% | **~5–10%** |
| **60% (our 2hr target)** | **10%** | **~25%** | **~35–45%** | **~30–40%** |
| 80% | 37% | ~60% | ~75% | **~65–75%** |
| 90% | 62% | ~80% | ~90% | **~80–85%** |

**How dictionary and LLM constraints help:**
- **Dictionary constraint**: the phone sequence "K-AE-T" can only map to "CAT" (and a handful of other words). Even if one phone is wrong, the dictionary narrows candidates dramatically. A 40K-word English dictionary constrains phone sequences to far fewer options than unconstrained phone permutations.
- **LLM context**: "The ___ sat on the mat" — even with no phone information, the LLM knows this is likely "cat", "dog", or "boy". Combined with a noisy phone lattice showing K/G in position 1 and AE/EH in position 2, "CAT" becomes overwhelmingly likely.

### 7.3 Predicted Free Speech Performance

#### Scenario A: After 2-hour calibration (our plan)

| Metric | Estimate | Confidence |
|---|---|---|
| Phone top-1 accuracy | 30–45% | Medium |
| Phone top-5 accuracy | 60–75% | Medium |
| **Raw word accuracy** (no LLM) | 5–15% | Low |
| **Word accuracy with LLM** | **25–40%** | Medium |
| **Content word recovery** | **20–35%** | Medium |
| **Sentence gist recovery** | **30–50%** | Medium |
| WER | 60–90% | Medium |

"Sentence gist recovery" means a human reading the LLM output can understand the general topic/intent of the original sentence, even if specific words are wrong.

#### Scenario B: Extended calibration (4+ hours, across sessions)

| Metric | Estimate | Confidence |
|---|---|---|
| Phone top-1 accuracy | 45–60% | Medium |
| Phone top-5 accuracy | 75–85% | Medium |
| Word accuracy with LLM | 40–60% | Medium |
| Content word recovery | 35–55% | Medium |
| WER | 40–65% | Medium |

#### Scenario C: Research-grade (20+ hours, DNN, parallel audio)

This is Gaddy (2021) territory:

| Metric | Published result | With modern LLM |
|---|---|---|
| WER (silent speech) | 68% | **Est. 40–55%** |
| WER (vocalized) | 4% | ~2% |

The 13–28pp improvement estimate from adding a modern LLM is based on published LLM error correction results showing 10–54% relative WER improvement.

### 7.4 Constrained Vocabulary: The Sweet Spot

For practical hackathon use, the most impactful configuration is **constrained-vocabulary LLM decoding**:

- Define a vocabulary of 50–200 task-relevant words
- Phone lattice + dictionary constraint → candidate words per position
- LLM selects from candidates using context

| Vocabulary size | Phone top-5 needed | Estimated word accuracy |
|---|---|---|
| 8 words (commands) | 40% | 75–90% |
| 50 words | 50% | 50–70% |
| 200 words | 60% | 35–55% |
| 1000 words | 70% | 25–40% |
| Open (40K+) | 85%+ | 15–30% |

**Recommendation:** For the hackathon demo, use a **50–200 word constrained vocabulary** matched to the application domain (e.g., browser commands, navigation terms, common phrases). This gives the LLM enough constraint to produce useful output even with moderate phone accuracy.

---

## 8. Mathematical Derivations

### 8.1 Phone-to-Word Probability

Let `p₅` = phone top-5 accuracy (probability the correct phone is in top-5 candidates).

For a word with `n` phones, the probability that ALL phones are covered in their top-5:

```
P(word covered) = p₅ⁿ
```

Average English word: `n ≈ 4.5`

| p₅ | P(word covered) | Words/sentence (8 words) covered |
|---|---|---|
| 0.40 | 1.6% | 0.13 words |
| 0.60 | 10.0% | 0.8 words |
| 0.70 | 19.2% | 1.5 words |
| 0.80 | 36.6% | 2.9 words |
| 0.90 | 62.2% | 5.0 words |

### 8.2 Dictionary Constraint Boost

With a vocabulary of `V` words, the average number of dictionary entries matching a given phone position's top-5 is roughly:

```
candidates_per_position ≈ V × (5/41)ⁿ
```

For V=1000, n=4.5: candidates ≈ 1000 × 0.122⁴·⁵ ≈ 1000 × 0.00015 ≈ 0.15

This means with 1000 words and top-5 phones, most words have **0-1 dictionary matches** — the dictionary itself is a powerful disambiguator. Even if not all phones are covered, partial phone matches + dictionary lookup dramatically narrow the search.

### 8.3 LLM Context Boost

Based on ASR literature, language model rescoring provides:
- **N-gram LM**: 20–30% relative WER reduction
- **Neural LM**: 30–40% relative WER reduction
- **LLM (GPT-class)**: 40–54% relative WER reduction (published 2024 results)

Applied to our estimates:
```
WER_with_LLM = WER_raw × (1 - LLM_reduction)
```

Conservative LLM reduction = 35%, aggressive = 50%.

### 8.4 Data Yield from 2-Hour Session

```
Phase 1 (Commands): 8 × 75 = 600 command segments
Phase 2 (Words):    600 words × 4.5 phones = 2,700 phone segments
Phase 3 (Sentences): 450 sentences × 8 words × 4.5 phones = 16,200 phone segments

Total phone segments: ~18,900
Per phone class (41): ~461 on average
    Common phones (T,N,S): ~800-1200
    Rare phones (ZH,OY):  ~50-150
```

### 8.5 Accuracy Scaling Model

Based on fitting to known data points (AlterEgo, our experiments, literature), command classification accuracy follows an approximate logistic curve:

```
acc(n) = chance + (ceiling - chance) / (1 + exp(-k × (n - n_mid)))
```

Where:
- `n` = samples per class
- `chance` = 1/n_classes
- `ceiling` ≈ 0.96 - 0.005 × n_classes (diminishes with more classes)
- `k` ≈ 0.08 - 0.003 × n_classes
- `n_mid` ≈ 15 + 1.5 × n_classes

This model predicts:
- 8 classes, 75 samples: **89%** (matches AlterEgo's 92% on 10 classes)
- 5 classes, 50 samples: **85%** (consistent with our adjusted EMG-UKA extrapolation)

---

## 9. Session Planning: The 2-Hour Calibration

### 9.1 Minute-by-Minute Schedule

| Time | Activity | Notes |
|---|---|---|
| 0:00–0:05 | Hardware setup, electrode placement | Mark positions, check impedance |
| 0:05–0:10 | Signal quality check | Jaw clench, tongue, swallow tests |
| 0:10–0:15 | Practice round | 5 samples each of 3 commands, discard data |
| 0:15–0:45 | **Phase 1: Command calibration** | 75 × 8 = 600 samples |
| 0:45–0:50 | Break + quick model check | Train RF, display accuracy |
| 0:50–1:20 | **Phase 2: Phone drilling** | 600 words → ~2,700 phone segments |
| 1:20–1:25 | Break + electrode check | Re-seat if needed, mark positions |
| 1:25–2:15 | **Phase 3: Free speech** | 450 sentences → ~16,200 phone segments |
| 2:15–2:25 | Final model training + eval | Train all classifiers, report metrics |

### 9.2 Session Independence

**Critical:** collect data across at least 2 sub-sessions with a break between them. Electrode micro-shifts during the break simulate real-world usage conditions.

Split for training:
- Phase 1 first half + Phase 3 first half → training set
- Phase 1 second half + Phase 3 second half → validation set
- Or use the break as a natural session boundary

### 9.3 Fatigue Management

Silent speech is surprisingly tiring. After ~20 minutes of continuous vocalization, users report:
- Mental fatigue (reduced consistency)
- Jaw/tongue fatigue (altered muscle patterns)
- Attention drift (less precise vocalization)

**Mitigations:**
- Enforce breaks every 20–30 minutes
- Vary the task (commands → words → sentences) to maintain engagement
- Keep sessions to <45 minutes of continuous collection
- Provide visual progress bar and encouragement

### 9.4 Real-Time Quality Monitoring

During collection, monitor and flag:
- **Dead channels** (electrode disconnected): flat signal
- **Movement artifacts** (large spikes): reject and re-collect
- **Inconsistent segments** (wildly different duration): may indicate mis-vocalization
- **Running accuracy** (after 30+ samples): should trend upward

---

## 10. Failure Modes & Contingencies

### 10.1 Signal Quality Failures

| Problem | Detection | Fix |
|---|---|---|
| No signal on a channel | Flat line in raw data | Check electrode, re-apply gel |
| 60 Hz noise dominates | Visible oscillation | Check ground electrode, move away from power sources |
| Movement artifacts | Large random spikes | Instruct user to stay still, secure electrodes better |
| All channels look same | No differential response | Electrode placement too close; spread out |

### 10.2 Accuracy Failures

| Problem | Detection | Likely cause | Fix |
|---|---|---|---|
| Command acc <50% | Phase 1 checkpoint | Poor electrode contact | Re-do electrode placement |
| Command acc 50–70% | Phase 1 checkpoint | Phonetically similar commands | Replace confusing commands |
| Phone acc <15% top-1 | Phase 2 eval | Insufficient data or signal | Focus on manner classes instead |
| LLM outputs gibberish | Phase 3 eval | Phone acc too low | Fall back to command mode |

### 10.3 Fallback Strategy

If full-pipeline accuracy is disappointing:

1. **Tier 1 fallback**: Reduce to 5 commands instead of 8 → +10pp accuracy
2. **Tier 2 fallback**: Use only manner-of-articulation (6 classes) instead of phones → more reliable for LLM
3. **Tier 3 fallback**: Command-only mode (abandon free speech) → guaranteed 80%+ with good signal
4. **Tier 4 fallback**: Binary classification (yes/no or on/off) → near-certain to work

### 10.4 The "Demo Tax"

In live demos, expect **5–10pp accuracy drop** vs. calibration results due to:
- User nervousness (altered muscle tension)
- Time pressure (less precise vocalization)
- Electrode shift from extended wear
- Environmental noise (electrical interference)

**Plan for this:** if calibration gives 85%, demo will likely show ~75–80%. Set demo expectations accordingly.

---

## 11. Quick-Start Checklist

### Pre-Session (Day Before)
- [ ] Test all electrodes with multimeter (resistance < 10 kΩ)
- [ ] Verify ADC connection and sample rate (250–600 Hz)
- [ ] Prepare word lists for Phase 2 (covering all 41 phones)
- [ ] Prepare sentence lists for Phase 3 (500 sentences, phonetically balanced)
- [ ] Pre-compute CMU dict phone sequences for all Phase 2/3 words
- [ ] Test full software pipeline with mock data (end-to-end)
- [ ] Charge any wireless hardware, prepare spare electrodes

### Setup (Minutes 0–10)
- [ ] Clean skin at electrode sites with alcohol
- [ ] Apply electrodes with conductive gel in priority order
- [ ] Connect to ADC, verify all channels show signal
- [ ] Run signal quality check (jaw clench, tongue, swallow)
- [ ] Mark electrode positions with skin marker

### Phase 1 — Commands (Minutes 15–45)
- [ ] Start command collection UI
- [ ] Collect 30 samples/command → checkpoint accuracy
- [ ] If any command <40% accuracy, replace it
- [ ] Complete to 75 samples/command
- [ ] Train RF model, record accuracy

### Phase 2 — Phone Drill (Minutes 50–80)
- [ ] Load word list, start phone collection UI
- [ ] Collect 600 isolated words
- [ ] Monitor for fatigue, enforce pace

### Phase 3 — Free Speech (Minutes 85–135)
- [ ] Re-check electrodes after break
- [ ] Load sentence list, start sentence collection UI
- [ ] Collect 400–450 sentences
- [ ] Monitor signal quality throughout

### Post-Session (Minutes 135–145)
- [ ] Train command classifier → report accuracy
- [ ] Train phone classifier → report top-1 and top-5 accuracy
- [ ] Train manner classifier → report accuracy
- [ ] Run LLM decoding on 10 held-out sentences → report WER
- [ ] Save all models and data

### Expected Outputs

| Deliverable | Expected metric |
|---|---|
| Command classifier (8 commands) | 82–90% accuracy |
| Phone classifier (41 classes) | 30–45% top-1, 60–75% top-5 |
| Manner classifier (6 classes) | 60–75% accuracy |
| LLM free speech (constrained vocab) | 25–40% word accuracy |
| LLM free speech (open vocab) | 15–25% word accuracy |

---

## Appendix A: Comparison of Our Results to Literature

| Metric | Our EMG-UKA (worst case) | Our predicted (2hr) | AlterEgo | Gaddy (20hr) | SOTA 2025 |
|---|---|---|---|---|---|
| **Task** | Continuous, extracted | PTT isolated | PTT isolated | Continuous | Varies |
| **Data** | ~50 samples/phone | ~450/phone | 75/class | 20 hours | Varies |
| **Command acc (8 cls)** | 44% | **85%** | 92% (10 cls) | N/A | 96% |
| **Phone top-1 (41 cls)** | 5.7% | **35%** | N/A | ~55% (est) | N/A |
| **Open vocab WER** | >100% | **60–90%** | N/A | 68% (silent) | N/A |
| **Architecture** | RF | RF | 1D CNN | Enc-Dec DNN | Various |

## Appendix B: Glossary

| Term | Definition |
|---|---|
| **PTT** | Push-to-talk — user holds button during vocalization for clean segmentation |
| **WER** | Word Error Rate — (substitutions + insertions + deletions) / reference words |
| **Top-k** | Model outputs k most likely classes with probabilities |
| **Manner** | Articulatory manner — how a sound is produced (stop, fricative, vowel, etc.) |
| **Phone/Phoneme** | Smallest unit of speech sound (English has ~41) |
| **MFCC** | Mel-Frequency Cepstral Coefficients — spectral features standard in speech recognition |
| **Coarticulation** | Influence of adjacent sounds on each other's articulation |
| **Forced alignment** | Using known transcript + phone dictionary to estimate phone boundaries |

---

*Document based on experimental results from EMG-UKA corpus evaluation, AlterEgo paper (Kapur et al. 2018), Gaddy & Klein (EMNLP 2020, ACL 2021), Meltzner et al. (2011), Wand & Schultz (2014, 2016), and 2024–2025 sEMG silent speech literature. Accuracy estimates combine empirical results with mathematical modeling and published benchmarks.*
