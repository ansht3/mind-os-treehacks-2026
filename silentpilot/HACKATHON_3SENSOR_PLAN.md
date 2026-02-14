# SilentPilot: 3-Sensor Hackathon Build Plan

## What We Have, What We Can Build, What to Expect

---

## The Hardware

**3 × MyoWare 2.0 Muscle Sensors**
- Analog output: 0–3.3V (ENV mode = smoothed envelope, RAW mode = amplified EMG)
- Onboard gain: 200× base, adjustable via potentiometer
- Bandpass: 20.8–498.4 Hz built-in (good — captures speech EMG range)
- Each sensor has 3 snap electrodes: MID (center of muscle), END (along muscle), REF (nearby bone/non-active area)

**ESP32 (from our existing firmware)**
- 4 ADC pins available (GPIO 32, 33, 34, 35) — we use 3
- 12-bit ADC, 250 Hz sample rate
- Binary packet protocol already implemented in `firmware/emg_streamer/`
- Serial → Python pipeline already implemented in `emg_core/ingest/serial_reader.py`

**What this means:** The software stack is ready. We connect 3 MyoWare outputs to 3 ESP32 ADC pins and the entire pipeline from hardware → features → classifier → WebSocket → UI works out of the box.

---

## Sensor Placement: Where to Put the 3 Sensors

With only 3 sensors, every placement decision matters. Here's the optimal configuration, ranked by information value for silent speech:

```
         FRONT VIEW                    SIDE VIEW

         ┌───────┐                        │
         │ FACE  │                     ┌──┤
         │       │                     │  │
         │       │                     │  │  ← [1] Jaw (Masseter)
    [1]→ │  ██   │                     │  │     Right side of jaw
         │       │                     │  │
         └───┬───┘                     └──┤
             │                            │
        ┌────┴────┐                    ┌──┤
        │  CHIN   │  ← [2]            │  │  ← [2] Under chin (Submental)
        └────┬────┘                    │  │     Centered below jawline
             │                         └──┤
        ┌────┴────┐                       │
        │ THROAT  │  ← [3]               │  ← [3] Throat (Laryngeal)
        │         │                       │     Over the larynx/Adam's apple
        └─────────┘                       │
```

### Sensor 1: MASSETER (Jaw) — GPIO 32

**Location:** Right cheek, over the jaw muscle. Place your fingers on your cheek and clench your teeth — the muscle that bulges is the masseter.

**What it captures:**
- Jaw opening/closing (distinguishes vowel heights)
- Bite force (distinguishes stops like T/D/K from fricatives like S/F)
- General "speech is happening" activation

**Electrode placement:**
- MID electrode: center of the masseter (where it bulges most when clenching)
- END electrode: toward the jaw hinge (near the ear)
- REF electrode: on the cheekbone (zygomatic arch) — bony, minimal muscle

**MyoWare physical fit:** Masseter is large enough for the MyoWare board. Orient the board vertically along the jaw.

### Sensor 2: SUBMENTAL (Under Chin) — GPIO 33

**Location:** Centered under the chin, on the soft tissue between the jawbone. This captures tongue and hyoid bone movement — the **single most informative location** for speech.

**What it captures:**
- Tongue elevation (T, D, N, L vs. K, G, NG)
- Tongue root movement (front vowels IY/EH vs. back vowels UW/AO)
- Swallowing and hyoid elevation

**Electrode placement:**
- MID electrode: dead center under the chin
- END electrode: slightly forward toward the chin point
- REF electrode: on the underside of the jawbone (bony surface)

**MyoWare physical fit:** This is tight. The submental space is small. **Tip:** Use electrode lead cables (if available) to place the snap electrodes under the chin while mounting the MyoWare board itself on the neck or chest. If no lead cables, carefully tape the board under the chin — it will work but may be uncomfortable.

### Sensor 3: LARYNGEAL (Throat) — GPIO 34

**Location:** Over the larynx (Adam's apple area). This captures vocal fold and laryngeal muscle activity.

**What it captures:**
- Voicing distinction (P vs. B, T vs. D, S vs. Z)
- Laryngeal tension during speech vs. silence
- Word boundary detection (activation onset/offset)

**Electrode placement:**
- MID electrode: just above the Adam's apple
- END electrode: just below the Adam's apple
- REF electrode: on the side of the neck over the sternocleidomastoid (the big neck muscle)

**MyoWare physical fit:** Good — the throat provides a flat surface. Orient vertically.

### What We Lose Without Sensors 4–6

| Missing location | What we lose | Impact |
|---|---|---|
| Left masseter | Bilateral jaw info | Minor — one side captures most info |
| Buccal (cheek) | Lip rounding, cheek tension | Moderate — OW/UW vs. IY/IH harder |
| Infrahyoid (lower throat) | Secondary laryngeal detail | Minor — sensor 3 covers primary |

**Net impact on accuracy:** Expect approximately **5–15pp drop** compared to a 6-channel system, depending on the task. Commands that differ mainly in lip shape (e.g., "OPEN" vs. "EAST") will be harder. Commands that differ in tongue/jaw/voicing (e.g., "STOP" vs. "GO") will be almost as accurate.

---

## Signal Setup: ENV vs. RAW Mode

### Use ENV (Envelope) Mode

**Why:** ENV gives a smooth muscle activation signal — exactly what our feature extraction expects. The MyoWare's built-in rectifier + low-pass filter at 3.6 Hz produces a clean amplitude envelope that maps directly to "how hard is this muscle working right now."

Our `extract_features()` function computes RMS, MAV, waveform length, etc. — these all work best on envelope signals. MFCCs will have lower spectral resolution on the envelope, but with only 250 Hz sample rate and 3 channels, the time-domain features will likely dominate anyway.

### Gain Adjustment

**Start with the potentiometer at midpoint.** Then:
1. Have the user silently say "EEE" (tongue up, jaw semi-closed)
2. Watch the analog values — should swing between ~500–3000 (out of 4095)
3. If signal is <200, turn gain UP
4. If signal is >3800 (clipping), turn gain DOWN
5. The sweet spot: silent rest reads ~200–500, active speech reads ~1500–3500

---

## Wiring

```
MyoWare Sensor 1 (Jaw)     →  ENV out → GPIO 32  (ADC1_CH4)
MyoWare Sensor 2 (Chin)    →  ENV out → GPIO 33  (ADC1_CH5)
MyoWare Sensor 3 (Throat)  →  ENV out → GPIO 34  (ADC1_CH6)

All sensors:
  VIN → ESP32 3.3V
  GND → ESP32 GND

(GPIO 35 / ADC1_CH7 unused — leave disconnected)
```

### Firmware Change

The firmware at `firmware/emg_streamer/emg_streamer.ino` is configured for 4 channels. We need to update it to use 3:

```cpp
// Change from:
#define NUM_CHANNELS 4
const int adcPins[NUM_CHANNELS] = {32, 33, 34, 35};

// Change to:
#define NUM_CHANNELS 3
const int adcPins[NUM_CHANNELS] = {32, 33, 34};
```

Also update the packet parser in `emg_core/ingest/packet_parser.py` and `config.py`:

```python
# config.py
NUM_CHANNELS = 3  # was 4
```

### Feature Count

With 3 channels, our feature vector changes:
- Time-domain: 5 features × 3 channels = 15
- MFCCs: 16 features × 3 channels = 48  
- Cross-channel ratios: 3 (all pairs of 3 channels)
- **Total: 66 features** (was 87 for 4 channels, 129 for 6)

This is still plenty for RF classification.

---

## What We Can Realistically Build

### Tier 1: Command Classifier (DEFINITELY DO THIS)

**5 commands, 85–90% accuracy. Ready in 30 minutes of calibration.**

This is our bread and butter and the foundation of every demo.

**Recommended 5-command set for 3 sensors:**

| Command | Why it works with jaw/chin/throat |
|---|---|
| **STOP** | Strong fricative (S) + jaw drop (AA) — big signal on all 3 |
| **GO** | Velar stop (G) + back vowel (OW) — strong throat + chin |
| **YES** | Palatal onset (Y) + front vowel (EH) + fricative (S) — distinct chin pattern |
| **NO** | Nasal (N) + back rounded vowel (OW) — strong chin + throat |
| **CLICK** | Velar stop (K) + liquid (L) — distinct jaw pattern |

These 5 were chosen specifically because they differ primarily in **tongue position, jaw openness, and voicing** — exactly what our 3 sensors capture best. We deliberately avoid commands that differ mainly in lip shape (we can't see lips).

**What to avoid:** Pairs like OPEN/OWN (differ in lip rounding), FIVE/FINE (differ subtly), BEAT/BEET (near-identical articulation).

**Expected accuracy:**

| Samples/command | 5 commands | Notes |
|---|---|---|
| 25 | 60–70% | Quick test — enough to validate signal |
| 50 | 75–85% | Solid demo with occasional errors |
| **75** | **82–90%** | **Target — reliable for live demo** |
| 100 | 85–92% | Diminishing returns |

### Tier 2: Extended Commands (8 commands)

If Tier 1 accuracy is >85%, extend to 8 commands:

| Command | Articulatory signature |
|---|---|
| STOP | Fricative + stop + back vowel |
| GO | Velar + back vowel |
| YES | Palatal + front vowel + fricative |
| NO | Nasal + diphthong |
| CLICK | Velar stops bracket vowel |
| SCROLL | Fricative + cluster + back vowel + lateral |
| BACK | Bilabial + front vowel + velar |
| NEXT | Nasal + front vowel + cluster |

**Expected accuracy: 70–82%** with 75 samples each.

### Tier 3: Phrase Classification (15–20 phrases)

Map each phrase to a single classification target:

```
"Call 911"        "I need help"      "Send location"
"Open browser"    "Search for"       "Go back"
"Scroll down"     "Click that"       "Stop"
"Yes"             "No"               "Cancel"
```

**Expected accuracy: 55–72%** — usable with a confirmation step, but risky for live demo without it.

### Tier 4: Phone Classification → LLM (STRETCH GOAL)

With 3 channels, phone classification will be **significantly degraded**:
- Top-1: 15–25% (41 phones) — probably too low for useful LLM decoding
- Top-5: 35–50%
- Manner (6 classes): 40–55%

**Honest assessment:** Open-vocabulary LLM decoding with 3 sensors is unlikely to produce usable sentences. The phone lattice will be too noisy. But it could work as a "research preview" segment of the demo if we constrain to <50 words and pre-practice sentences.

---

## Data Collection Plan: The 45-Minute Session

We don't have 2 hours. Hackathon reality means we want a usable model in under an hour.

### Quick Schedule

| Time | Activity | Output |
|---|---|---|
| 0:00–0:10 | Wire sensors, place electrodes, check signal | Working hardware |
| 0:10–0:12 | Signal quality check (clench, tongue, swallow) | Verified all 3 channels |
| 0:12–0:15 | Practice round (5 samples per command, discard) | User learns the rhythm |
| 0:15–0:35 | **Collect 75 samples × 5 commands** = 375 samples | Phase 1 data |
| 0:35–0:37 | Train RF model, check accuracy | Checkpoint |
| 0:37–0:42 | **If >80%:** collect 3 more commands (75 × 3) | Phase 2 data |
| 0:42–0:45 | Final model training + evaluation | Deployable model |

**Total active collection time:** ~25 minutes for 5 commands, ~35 minutes for 8.

### PTT Protocol

Our existing UI supports PTT calibration via the WebSocket API:
1. Frontend sends `calib_start` with command label
2. User silently vocalizes for ~1 second
3. Frontend sends `calib_stop`
4. Backend segments and stores the sample
5. Repeat

The 150-sample segment at 250 Hz = **0.6 seconds** per sample. With a 2-second gap between samples, each sample takes ~3 seconds total.

### If Short on Time: The 15-Minute Emergency Plan

| Time | Activity | Output |
|---|---|---|
| 0:00–0:05 | Wire + signal check | Working hardware |
| 0:05–0:15 | 30 samples × 5 commands = 150 samples | Minimal model |

**Expected accuracy: 65–75%** — rough but enough to demonstrate the concept is real.

---

## What to Expect: Honest Accuracy Predictions

### The Optimistic-But-Realistic View

| Scenario | Commands | Samples | Expected accuracy | Demo-ready? |
|---|---|---|---|---|
| **Best case** | 5 | 100 | 88–92% | Absolutely |
| **Likely** | 5 | 75 | 82–88% | Yes |
| **Acceptable** | 8 | 75 | 70–80% | With confirmation UX |
| **Tight timeline** | 5 | 30 | 60–72% | Risky but possible |
| **Disaster** | Any | Any | <50% | Signal problem, re-place electrodes |

### The 3-Sensor Penalty

Compared to the 6-channel estimates in our full DATA_CAPTURE_PLAN.md, expect:

| Task | 6-channel estimate | 3-channel estimate | Penalty |
|---|---|---|---|
| 5 commands @ 75 samples | 88–93% | 82–88% | -6pp |
| 8 commands @ 75 samples | 82–90% | 70–80% | -10pp |
| Phone top-1 (41 cls) | 30–45% | 15–25% | -15pp |
| Phone top-5 (41 cls) | 60–75% | 35–50% | -20pp |
| Manner (6 cls) | 60–75% | 40–55% | -17pp |

The penalty grows with task difficulty because harder tasks rely on subtle inter-channel differences that the missing sensors would have provided.

### What 3 Sensors Are Good At

The jaw/chin/throat trio excels at distinguishing:
- **Open vs. closed jaw:** "AH" vs. "EE" — big signal difference on masseter
- **Tongue front vs. back:** "T" vs. "K" — clear submental difference
- **Voiced vs. unvoiced:** "B" vs. "P" — laryngeal sensor lights up for voiced
- **Speech vs. silence:** all 3 channels activate during speech — reliable VAD

### What 3 Sensors Are Bad At

- **Lip rounding:** "OO" vs. "EE" look similar without cheek sensors
- **Bilabial details:** "M" vs. "N" are hard (both nasal, differ in lip closure)
- **Whispered subtlety:** very quiet subvocalization may not register on ENV mode

---

## Recommended Demo With 3 Sensors

Given the constraints, here's what we should actually demo:

### Primary Demo: Silent Browser Control (Demo 5 from DEMO_IDEAS.md)

**Why this one:**
- Only needs 5–8 commands → our sweet spot with 3 sensors
- Visually immediate — audience sees actions happening
- Integrates with our existing MCP browser control
- Doesn't require perfect accuracy (a retry is fine)

**Commands:**

| Silent command | Browser action |
|---|---|
| "SCROLL" | Scroll down |
| "BACK" | Navigate back |
| "CLICK" | Click highlighted element |
| "NEXT" | Highlight next element |
| "STOP" | Cancel / stop scrolling |

5 commands. 75 samples each. 20 minutes to calibrate. **82–88% expected accuracy.**

### Secondary Demo: Silent Emergency Text (Demo 3)

If command accuracy is >80%, add a second scene with emergency phrases. Even with just 3 more commands added to the 5 above:

| Silent command | Action |
|---|---|
| "HELP" | Send "I NEED HELP" text via Twilio |
| "POLICE" | Send "SEND POLICE" text |
| "SAFE" | Send "I'M SAFE" text |

8 total commands. **70–80% expected accuracy** — use a confirmation step ("did you mean HELP? Say YES to confirm") to bring effective accuracy to ~95%.

### Stretch: Silent AI Query

If there's time left, pre-train 5 question templates:

| Silent command | Query to GPT |
|---|---|
| "WEATHER" | "What's the weather?" |
| "TIME" | "What time is it?" |
| "NEWS" | "Summarize today's news" |
| "JOKE" | "Tell me a joke" |
| "TRANSLATE" | "Translate my last text to Spanish" |

These are just 5 more commands added to the classifier (13 total). At this count, expect **60–70% raw accuracy**, but the errors are entertaining rather than catastrophic (asking for a joke instead of weather is funny, not broken).

---

## Troubleshooting Quick Reference

| Symptom | Likely cause | Fix |
|---|---|---|
| All channels read ~0 | No power to sensors | Check 3.3V connection |
| All channels read ~4095 | Saturated / gain too high | Turn down MyoWare potentiometer |
| One channel flat | Electrode disconnected | Re-snap electrode, check gel |
| Signal jumps wildly | Loose electrode or movement | Secure with tape, hold still |
| Accuracy <50% on 5 commands | Poor electrode contact or placement | Redo electrode placement from scratch |
| 2 commands always confused | Articulatorily too similar | Replace one of the pair with a different command |
| Training accuracy 95% but test 40% | Overfitting (too few samples) | Collect more samples, reduce features |
| ESP32 not sending data | Serial port wrong | Check `ls /dev/tty*`, update SERIAL_PORT |
| Packets corrupted | Baud rate mismatch | Ensure both sides use 115200 |

---

## Summary: The Honest Take

**With 3 MyoWare sensors, we can reliably build a 5–8 command silent speech interface that works at 75–88% accuracy.** This is enough for a compelling live demo of silent browsing, silent emergency communication, or silent AI interaction.

What we **cannot** reliably do with 3 sensors:
- Open-vocabulary free speech (need 6+ channels and vastly more data)
- Fine-grained phone classification (too few channels for lip/cheek info)
- More than ~15 distinct commands at usable accuracy

**The play:** Nail the 5-command demo first. If it works great, extend to 8. If that works, add the emergency scenario. Don't overreach — a polished 5-command demo at 88% beats a shaky 15-command demo at 55%.

**Time budget:** Hardware setup (10 min) + calibration (25 min) + training (2 min) = **model ready in 37 minutes**. This leaves plenty of hackathon time for the UI, demo polish, and practice runs.
