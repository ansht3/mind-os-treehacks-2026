# MindOS — TreeHacks 2026

MindOS is a non-invasive silent speech interface that converts micro muscle signals (EMG) from the jaw and throat into real computer actions.  
It enables users to communicate and control software without speaking or using their hands.

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/v-DJEQgKm-A/maxresdefault.jpg)](https://www.youtube.com/watch?v=v-DJEQgKm-A)

---

## 💡 Inspiration

Imagine having a perfectly clear mind but being unable to communicate it to the world. Millions of people living with ALS, recovering from stroke, or experiencing severe paralysis cannot rely on keyboards or speech interfaces. Existing tools fail exactly where assistive technology matters most.

We built MindOS to create a non-invasive interface that translates silent intent into digital action — providing a pathway for communication and control without implants, speech, or physical movement.

---

## What it Does

- **Silent intent → action**  
  Users silently mouth commands detected through EMG signals

- **Hands-free computer control**  
  Browse, search, scroll, and navigate without a keyboard or voice

- **Adaptive learning loop**  
  Users can append new training data to personalize decoding

- **Assistive-first design**  
  Built for people who cannot rely on speech or motor input

---

## How It Works

MindOS is a modular pipeline that connects biological signals to digital actions.

### 1. Hardware & Signal Ingestion
- Two **MyoWare EMG muscle sensors** placed along the jaw
- Signals routed through an **Arduino Uno**
- Real-time streaming via **PySerial** into the processing pipeline

### 2. Signal Processing
- Treated EMG as **time-series data**
- Applied noise filtering to remove motion artifacts
- **Random Forest classifier** maps signals to four biometric phoneme classes

### 3. Multi-Agent Decision Layer

Because EMG signals are low-entropy, we designed a structured agent workflow:

- **Context Agent**  
  Infers likely characters/words using linguistic priors

- **Action Agent**  
  Determines which computer command the user intends

- **Execution Agent**  
  Uses **Playwright** to perform browser automation safely

### Reliability Layer
A plug-and-play signal interface allows switching between:
- Live sensor input  
- Mock signals for deterministic demos  

---


**Key Components**

- EMG signal ingestion via PySerial  
- Random Forest classification  
- Lexicon-filtered candidate generation  
- LLM-based contextual disambiguation  
- Tool-constrained agent execution  
- Browser automation with Playwright  

## Tech Stack

**Hardware**
- MyoWare EMG sensors  
- Arduino Uno  

**ML / Signal Processing**
- Python  
- Scikit-learn (Random Forest)  
- Time-series preprocessing  

**AI & Agents**
- GPT-4o (decision layer)  
- Custom context + action agents  

**Automation**
- Playwright  

**Frontend / Interface**
- Web UI for calibration + data collection  

---

## Challenges

### Limited Signal Resolution
With only two sensors, we could not build a full 26-class alphabet classifier.  
We were constrained to **four signal categories**.

---

## Accomplishments

- Built a full end-to-end assistive interface in a hackathon timeframe  
- Achieved **96% classification accuracy** on six hours of EMG data  
- Successfully controlled a live browser using silent intent  
- Designed a robust modular API for rapid iteration  

---

## What We Learned

- Real-world biosignals are noisy and highly user-dependent  
- Calibration and UX matter as much as model accuracy  
- Constrained agent design dramatically improves trust  
- Hardware-software co-design is essential for assistive tech  

---

## What’s Next

- Faster calibration for new users  
- Expanded command vocabulary  
- Multi-sensor fusion across facial muscle groups  
- On-device inference for lower latency and better privacy  
- Continuous learning from feedback logs  

---

## Why It Matters - Impact

We started this project because we know people personally that would want to work or even just be on their computer, but cannot. MindOS demonstrates that assistive computing doesn’t require invasive brain implants. By decoding neuromuscular signals already present during silent speech, we can create interfaces that restore autonomy and communication for millions of people.

---

## Acknowledgements

Thanks to the mentors, organizers, sponsors, and everyone else for their support at TreeHacks 2026.

---
