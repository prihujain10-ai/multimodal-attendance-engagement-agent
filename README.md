<div align="center">

<img src="https://img.shields.io/badge/-Classroom_Intelligence_Agent-1E3A5F?style=for-the-badge&logoColor=white" height="60" alt="Multimodal Classroom Agent Logo" />

# 🎓 Multimodal Attendance & Engagement Agent

**An AI-Powered, Privacy-Preserving Classroom Intelligence System**  
*Track sustained attendance, analyze real-time engagement dynamics, and empower educators with actionable pedagogical suggestions — completely offline.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Computer Vision](https://img.shields.io/badge/Computer_Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0+-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Audio DSP](https://img.shields.io/badge/Audio_DSP-VAD_%26_RMS-FF6F00?style=for-the-badge&logoColor=white)]()
[![Edge AI](https://img.shields.io/badge/Deployment-100%25_Offline_Edge-00C853?style=for-the-badge&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-2ECC71?style=for-the-badge)]()
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=for-the-badge)]()

<br/>

[🌟 Overview](#-what-is-the-multimodal-attendance--engagement-agent) • [🎭 Scenarios](#-the-three-core-scenarios) • [🏛️ Architecture](#️-system-architecture) • [🔬 Mathematical Fusion](#-mathematical-formulation-of-multimodal-fusion) • [🛠️ Tech Stack](#️-tech-stack) • [🔧 Installation](#-installation--quickstart) • [🛡️ Privacy](#-privacy-by-design--ethical-deployment) • [📊 Analytics](#-session-analytics--reports)

</div>

---

## 🌟 What is the Multimodal Attendance & Engagement Agent?

The **Multimodal Attendance & Engagement Agent** was engineered as an assistive, privacy-preserving classroom intelligence platform. Operating entirely on local edge hardware, the system integrates **computer vision**, **acoustic digital signal processing (DSP)**, and **agent-based heuristic reasoning** to:

1. **Eliminate roll calls:** Automatically confirm attendance based on sustained physical presence rather than momentary appearances.
2. **Quantify engagement:** Continuously fuse head pose orientation, body motion energy, inactivity duration, and classroom vocal dynamics into an objective engagement index $E(t)$.
3. **Assist educators in real time:** Identify attention drops, interaction peaks, and pacing stagnation to suggest non-enforcing pedagogical interventions (e.g., active learning pivots, formative checks, cold-calls).

> Designed from the ground up for **ethical deployment**: zero persistent raw video or audio is stored on disk, student identities are cryptographically pseudonymized with SHA-256 tokens, and the engine runs 100% offline without external cloud dependencies.

---

## ✨ The Three Core Scenarios

### 🎭 Scenario Comparison Matrix

| Dimension | 📋 Scenario 1: Automated Attendance Tracking | 📈 Scenario 2: Engagement Dynamics Analysis | 🤖 Scenario 3: AI-Driven Teaching Suggestions |
|:---|:---|:---|:---|
| **Primary Goal** | Flawless, automated roll call without human bias | Continuous telemetry of collective cognitive attention | Assistive pedagogical decision support for instructors |
| **Input Modalities** | Video stream (face detection & bounding boxes) | Multimodal: Head pose (yaw/pitch), optical motion, audio RMS | Aggregate interval metrics, trend derivatives, dynamic states |
| **Algorithmic Filter** | Temporal dwell debounce filter ($T_{\text{dwell}} \ge 15.0\text{s}$) | Moving window multimodal signal fusion: $E(t)$ | Heuristic cognitive decision tree & active learning rules |
| **False-Positive Defense** | Rejects hallway passersby and fleeting appearances | Distinguishes active note-taking from bored stagnation | Rate-limiting cooldowns to prevent teacher alert fatigue |
| **Output Artifact** | Timestamped registrar CSV roll call | Interval metrics, acoustic state, SVG trajectory curves | Actionable teaching prompts with urgency and rationale |

---

### 📋 Scenario 1: Automated Attendance Tracking
During a classroom session, the system processes a live or recorded video feed to detect and track participants:
- **Sustained Presence Verification:** Rather than single-frame detection (which flags passersby glancing into the room), attendance is only confirmed once an individual's cumulative presence crosses the configurable threshold ($T_{\text{dwell}} \ge 15.0\text{s}$).
- **Automated Roll Call:** Eliminates manual roll call, stops proxy attendance, and outputs timestamped registrar records.
- **Cryptographic Pseudonymization:** Preserves learner anonymity by converting visual identifiers into SHA-256 tokens (e.g., `STD-243E`).

---

### 📈 Scenario 2: Engagement and Session Dynamics Analysis
Classroom engagement cannot be measured by a single metric. The agent continuously synchronizes and fuses behavioral indicators across time intervals:
- **Head Pose & Gaze Orientation:** Measures yaw (horizontal turning) and pitch (slumping/looking down) relative to the instructor/whiteboard.
- **Movement Patterns & Inactivity:** Uses temporal frame differencing to quantify motion energy, penalizing long inactivity ($>45\text{s}$) and excessive agitation/restlessness ($>0.75$).
- **Acoustic Interaction Dynamics:** Evaluates RMS sound energy, voice activity detection (VAD), and speech-to-silence ratio to classify the acoustic state into `CONSTRUCTIVE_DIALOGUE`, `LECTURE_MONOLOGUE`, `PROLONGED_SILENCE`, or `HIGH_CHAOTIC_NOISE`.

---

### 🤖 Scenario 3: AI-Driven Teaching Suggestions
Acting as an assistive decision-support copilot, the agent evaluates interval dynamics to deliver contextual, non-enforcing pedagogical interventions:

```
┌───────────────────────────┬──────────────────────────┬────────────────────────────────────────────────────────────────────────┐
│ Detected Dynamic State    │ Pedagogical Rationale    │ Actionable Teaching Recommendation                                     │
├───────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ 🔻 Deep Attention Drop    │ Gaze fell <35% & stillness│ 💡 Rapid Engagement Pivot: Initiate a 60-second 'Think-Pair-Share'    │
│                           │ indicates passive fatigue│    or ask students to jot down one muddiest point on the board.        │
├───────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ ⏸️ Pacing Stagnation     │ Monologue audio & low    │ 💡 Pacing Adjustment: Shift from monologue to cold-calling or a quick  │
│                           │ motion across students   │    30-second thumbs-up/down confidence check on the worked example.    │
├───────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ ⭐ Interaction Peak       │ Robust dialogue & high   │ ⭐ Pedagogical Opportunity: High momentum! Synthesize contributions   │
│                           │ collective gaze (E > 80%)│    on the slide before moving into independent practice.               │
├───────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────────────────────┤
│ ⚡ Restlessness Spike     │ Agitated motion (>82%)   │ 💡 Focus Alignment: Refocus attention by posting an individual         │
│                           │ with low visual focus    │    challenge problem on the center board.                              │
└───────────────────────────┴──────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ System Architecture

```
[ Classroom Camera Feed ]                     [ Classroom Microphone Feed ]
            │                                               │
            ▼                                               ▼
  ┌─────────────────────────┐                     ┌─────────────────────────┐
  │      VisionTracker      │                     │     AudioProcessor      │
  │  - Face Detection       │                     │  - RMS Energy Extractor │
  │  - Head Pose (Yaw/Pitch)│                     │  - Adaptive Noise Floor │
  │  - Motion Energy Diff   │                     │  - Voice Activity (VAD) │
  │  - Inactivity Counter   │                     │  - Speech/Silence Ratio │
  └────────────┬────────────┘                     └────────────┬────────────┘
               │                                               │
               ├───────────────────────────┐                   │
               ▼                           ▼                   ▼
  ┌───────────────────────────┐    ┌─────────────────────────────────────────┐
  │ Scenario 1:               │    │ Scenario 2:                             │
  │ Sustained Presence Filter │    │ Multimodal Temporal Fusion Engine       │
  │ Dwell Time >= 15.0s       │    │ E(t) = wg*Gaze + wm*Motion + wa*Audio   │
  └────────────┬──────────────┘    └───────────────────┬─────────────────────┘
               │                                       │
               ▼                                       ▼
  ┌───────────────────────────┐    ┌─────────────────────────────────────────┐
  │ Attendance Roll Call      │    │ Scenario 3:                             │
  │ (Timestamped CSV)         │    │ AI Teaching Suggestion Agent            │
  └───────────────────────────┘    │ - Dynamic Pattern Classifier            │
                                   │ - Active Learning Decision Rules        │
                                   └───────────────────┬─────────────────────┘
                                                       │
                                                       ▼
                                   ┌─────────────────────────────────────────┐
                                   │ Analytics Reporter & Dashboard          │
                                   │ - Interactive HTML / SVG Trajectory     │
                                   │ - JSON Telemetry for LMS (Canvas/Moodle)│
                                   └─────────────────────────────────────────┘
```

---

## 🔬 Mathematical Formulation of Multimodal Fusion

The instantaneous engagement score $E(t) \in [0.0, 1.0]$ is computed as a weighted combination of three normalized multimodal streams:

$$E(t) = w_{\text{gaze}} \cdot \bar{V}_{\text{gaze}}(t) + w_{\text{motion}} \cdot \bar{V}_{\text{motion}}(t) + w_{\text{audio}} \cdot A_{\text{interaction}}(t)$$

Where the standard normalized weights satisfy:
$$w_{\text{gaze}} + w_{\text{motion}} + w_{\text{audio}} = 0.45 + 0.30 + 0.25 = 1.0$$

### 1. Visual Gaze Attention Score: $V_{\text{gaze}}(t)$
Quantifies deviation from the frontal line of sight ($\theta_{\text{yaw}} \le 25^\circ$, $\theta_{\text{pitch}} \le 20^\circ$):
$$V_{\text{gaze}}(t) = \max\left(0.0, 1.0 - \sqrt{\left(\frac{\max(0, |\theta_y| - 25^\circ)}{60^\circ}\right)^2 + \left(\frac{\max(0, |\theta_p| - 20^\circ)}{45^\circ}\right)^2}\right)$$

### 2. Physical Motion Score: $V_{\text{motion}}(t)$
Penalizes both prolonged stagnation ($t_{\text{inactivity}} > 45\text{s}$) and erratic fidgeting ($M > 0.75$):
$$V_{\text{motion}}(t) = \begin{cases} 
\max\left(0.05, 0.5 - 0.4 \cdot \frac{t_{\text{inact}} - 45}{30}\right) & \text{if } t_{\text{inactivity}} > 45\text{s} \\
\max\left(0.10, 0.7 - 0.4 \cdot \frac{M - 0.75}{0.25}\right) & \text{if } M > 0.75 \\
0.95 & \text{if } 0.10 \le M \le 0.60 \text{ (nominal active note-taking)} \\
0.80 & \text{otherwise}
\end{cases}$$

### 3. Acoustic Interaction Score: $A_{\text{interaction}}(t)$
Evaluates speech activity ratio $R_s$ over a 10-second sliding audio buffer:
$$A_{\text{interaction}}(t) = \begin{cases}
0.95 & \text{if } 0.15 \le R_s \le 0.70 \text{ (Constructive Q&A dialogue)} \\
0.70 & \text{if lecture monologue delivery} \\
0.40 & \text{if } R_s < 0.05 \text{ (Prolonged dead silence)} \\
0.50 & \text{if } R_s > 0.85 \text{ (Chaotic background noise)}
\end{cases}$$

---

## 🛠️ Tech Stack

```
Runtime                →  Python 3.10+ (Standard Library & NumPy)
Computer Vision        →  OpenCV (Haar Cascade / DNN / Optical Diff)
Signal Processing      →  NumPy (Vectorized RMS, VAD, Noise Floor EMA)
Audio Interface        →  SoundDevice / PyAudio (16kHz PCM Streaming)
Reasoning Agent        →  Heuristic Pedagogical Decision Engine
Analytics & Reporting  →  Python CSV, JSON, Self-Contained SVG/HTML5 Dashboard
CI & Automation        →  GitHub Actions Matrix (Python 3.10, 3.11, 3.12, 3.13)
```

---

## 🚀 Getting Started & Quickstart

### Prerequisites
- Python 3.10 or higher
- Git

### 🔧 Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/multimodal-attendance-engagement-agent.git
cd multimodal-attendance-engagement-agent

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt
```

---

## 💻 Running the System

### Option A: Run Offline Simulation (No Physical Hardware Needed)
To test and validate all 3 scenarios, run the built-in multi-phase session simulator:

```bash
python demo_session.py
```

**Simulation Workflow Demonstrated:**
- **Phase 1 (0–20s):** Concept Introduction — High frontal gaze, monologue delivery.
- **Phase 2 (20–40s):** Monotone Slump — Gaze drops to 25%, low motion, triggering **Scenario 3 Intervention**.
- **Phase 3 (40–60s):** Interactive Q&A — Collective speech and engagement surge to 97%.
- **Passerby Test:** An unregistered passerby enters between $t=5\text{s}$ and $t=9\text{s}$ (4s presence) and is **successfully rejected** by the sustained presence filter.

---

### Option B: Run with Live Classroom Camera & Microphone
When connected to an active USB webcam, RTSP classroom camera, or boundary microphone:

```bash
python live_camera_audio_runner.py
```

*Press `q` on the live video window to conclude the session and generate the intelligence reports.*

---

## 📋 Sample Simulation Cohort & Verification

| Participant | Type | Presence Observed | Sustained Dwell | Attendance Status | Pseudonym Assigned |
|:---|:---|:---:|:---:|:---:|:---:|
| `student_alpha` | Enrolled | 60.0s | 15.0s | ✅ **CONFIRMED** | `STD-243E` |
| `student_bravo` | Enrolled | 60.0s | 15.0s | ✅ **CONFIRMED** | `STD-39E0` |
| `student_charlie` | Enrolled | 60.0s | 15.0s | ✅ **CONFIRMED** | `STD-D0C1` |
| `student_delta` | Enrolled | 60.0s | 15.0s | ✅ **CONFIRMED** | `STD-096C` |
| `student_echo` | Enrolled | 60.0s | 15.0s | ✅ **CONFIRMED** | `STD-70F5` |
| `transient_visitor` | Passerby | 4.0s (t=5s to 9s) | 4.0s (< 15s) | ❌ **REJECTED (Passerby)** | *None (Discarded)* |

---

## 🗂️ Project Structure

```
multimodal_classroom_agent/
│
├── .github/
│   └── workflows/
│       └── ci.yml                     # Multi-Python GitHub Actions CI
│
├── config.py                          # Dwell thresholds, weights, and privacy toggles
├── vision_tracker.py                  # Face tracking, sustained presence filter, head pose
├── audio_processor.py                 # RMS energy, VAD, noise floor, acoustic state
├── fusion_engine.py                   # Multimodal temporal fusion & trend classifier
├── teaching_agent.py                  # Pedagogical decision engine & actionable suggestions
├── analytics_reporter.py              # Exporter for CSV, JSON, and interactive HTML
├── demo_session.py                    # Multi-phase end-to-end simulator
├── live_camera_audio_runner.py        # Physical webcam & microphone interface
│
├── outputs/                           # Generated session intelligence artifacts
│   ├── attendance_roll_call.csv       # Registrar-ready attendance log
│   ├── session_analytics.json         # Structured session trajectory for LMS
│   └── classroom_intelligence_report.html # Responsive SVG dashboard
│
├── .gitignore                         # Python/IDE exclusion rules
├── LICENSE                            # MIT License
├── requirements.txt                   # Dependency specifications
└── README.md                          # System documentation
```

---

## 🛡️ Privacy-by-Design & Ethical Deployment

This agent adheres to strict ethical AI and educational privacy standards:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRIVACY CERTIFIED SYSTEM                        │
├────────────────────────────────────────────────────────────────────────┤
│ 🛡️ 100% Offline Edge Computing: Zero telemetry sent to external APIs.  │
│ 🛡️ Zero Raw Media Persistence: Frames & audio processed in RAM only.  │
│ 🛡️ SHA-256 Pseudonymization: Raw biometric templates never stored.     │
│ 🛡️ Non-Coercive Decisions: Agent acts strictly as assistive support.  │
└────────────────────────────────────────────────────────────────────────┘
```

1. **In-Memory Streaming Buffers:** Video frames and audio chunks exist exclusively in ephemeral RAM buffers and are overwritten every cycle.
2. **Biometric Decoupling:** Facial coordinates and motion signals are converted into normalized scalar metrics; no facial photos or voice recordings are written to disk.
3. **FERPA / GDPR Compliance Friendly:** Data retained is limited to aggregate engagement indices and pseudonymized presence timestamps.

---

## 📊 Session Analytics & Reports

Each session outputs three production-ready artifacts:

### 1. Registrar Attendance CSV (`outputs/attendance_roll_call.csv`)
```csv
student_id,status,first_detected,confirmed_at,sustained_duration,timestamp_iso
STD-243E,PRESENT_CONFIRMED,1.0,16.0,15.0,2026-10-29 18:08:44
STD-39E0,PRESENT_CONFIRMED,1.0,16.0,15.0,2026-10-29 18:08:44
STD-D0C1,PRESENT_CONFIRMED,1.0,16.0,15.0,2026-10-29 18:08:44
```

### 2. LMS Telemetry JSON (`outputs/session_analytics.json`)
Structured interval metrics, trend slopes, and teaching interventions designed for direct ingestion into Canvas, Blackboard, or Moodle.

### 3. Interactive Visual Report (`outputs/classroom_intelligence_report.html`)
Standalone HTML5 dashboard featuring:
- Responsive SVG engagement timeline plot $E(t)$
- Summary cards (confirmed headcount, average engagement, peak flow windows)
- Audit log of AI pedagogical suggestions with urgency badges

---

## 🤝 Contributing

Contributions are warmly welcomed!

```bash
# 1. Fork the Project
# 2. Create your Feature Branch
git checkout -b feature/AdvancedGazeModel

# 3. Commit your Changes
git commit -m "Add: 3D head pose PnP estimation"

# 4. Push to the Branch
git push origin feature/AdvancedGazeModel

# 5. Open a Pull Request
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 👨‍💻 Authors & Acknowledgments

Developed as an open-source, multimodal classroom intelligence research system to empower educators and optimize student engagement through ethical, privacy-preserving AI.

> ⭐ **If you found this project helpful, please star this repository!**

---

<div align="center">

**Built with ❤️ for Privacy-First Educational Intelligence**

</div>
