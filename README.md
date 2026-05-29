# ClassGuard AI 🏫

> AI-powered automated teacher presence monitoring for classrooms.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Problem

When a teacher is temporarily absent, classrooms go unmonitored — leading to noise, indiscipline, and reduced learning efficiency. ClassGuard AI detects teacher absence in real time and automatically escalates alerts to the right people before the situation gets out of hand.

---

## ✨ Features

| Feature | Description |
|---|---|
| 👁️ **Teacher Detection** | Identifies enrolled teachers via face recognition |
| 🏃 **Motion Analysis** | Frame differencing detects excessive student movement |
| 🔊 **Noise Estimation** | Correlates motion patterns to approximate noise level |
| 🤖 **YOLOv8 Person Detection** | Counts people and identifies group formations |
| 📊 **Chaos Score** | Composite metric combining motion, noise, grouping, and standing behaviour |
| 🔔 **Escalating Alerts** | Teacher → Coordinator → Principal, on a configurable timer |
| 📈 **Analytics Dashboard** | Time-series charts of all metrics with CSV export |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI                        │
│  Live Monitor │ Enrollment │ Analytics │ Settings       │
└───────────────────────┬─────────────────────────────────┘
                        │
              ┌─────────▼──────────┐
              │  Detection Engine  │
              │  (detection_engine │
              │      .py)          │
              └──┬──────┬──────┬───┘
                 │      │      │
         ┌───────▼─┐ ┌──▼───┐ ┌▼──────────────┐
         │ YOLOv8  │ │ face │ │ OpenCV Frame  │
         │ Person  │ │ recog│ │ Differencing  │
         │Detection│ │nition│ │ (Motion/Noise)│
         └─────────┘ └──────┘ └───────────────┘
                        │
              ┌─────────▼──────────┐
              │   Alert Manager    │
              │  Teacher (60s)  →  │
              │  Coordinator(120s)→ │
              │  Principal (180s)  │
              └────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A webcam (or video file for testing)
- `cmake` installed on your system (required for `dlib`)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/classguard-ai.git
cd classguard-ai
```

**2. Install system dependencies**

```bash
# macOS
brew install cmake

# Ubuntu / Debian
sudo apt-get install cmake build-essential libopenblas-dev liblapack-dev

# Windows
# Install CMake from https://cmake.org/download/
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

> ⚠️ `dlib` takes 5–10 minutes to compile on first install. This is normal.

**4. Run the app**

```bash
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push your code to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Make sure **both** of these files are in the repo root:

**`requirements.txt`**
```
streamlit
numpy
pandas
opencv-python-headless
ultralytics
dlib
face_recognition
```

**`packages.txt`** ← required for `dlib` to compile on the cloud
```
cmake
build-essential
libopenblas-dev
liblapack-dev
libx11-dev
python3-dev
```

4. Deploy. The first build will take ~5 minutes due to `dlib` compilation.

---

## 📖 Usage

### 1. Enroll a Teacher

Go to **Teacher Enrollment** → upload a clear, front-facing photo → enter the teacher's name → click **Enroll**.

### 2. Start Monitoring

Go to **Live Monitor** → select your video source (webcam, uploaded video, or demo mode) → click **▶ Start**.

### 3. Alert Escalation

If the teacher is absent and the chaos score exceeds the threshold:

| Time elapsed | Action |
|---|---|
| 60 seconds | ⚠️ Reminder sent to teacher |
| 120 seconds | 🔴 Coordinator notified |
| 180 seconds | 🚨 Principal escalation |

Thresholds are configurable under **Settings**.

### 4. Analytics

View time-series charts of chaos score, motion, noise, and person count under **Analytics Dashboard**. Export alert logs as CSV.

---

## 🗂️ Project Structure

```
classguard-ai/
├── app.py                  # Main Streamlit entry point
├── requirements.txt
├── packages.txt            # System deps for Streamlit Cloud
│
├── pages/
│   ├── monitor.py          # Live monitoring feed
│   ├── enrollment.py       # Teacher face enrollment
│   ├── analytics.py        # Charts & alert log
│   ├── settings.py         # Configuration
│   └── about.py
│
└── utils/
    ├── detection_engine.py # YOLOv8 + face_recognition + motion
    ├── alert_manager.py    # Escalation state machine
    └── state.py            # Streamlit session state management
```

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io)** — UI framework
- **[YOLOv8](https://github.com/ultralytics/ultralytics)** — Real-time person detection
- **[face_recognition](https://github.com/ageitgey/face_recognition)** — Teacher identity verification
- **[OpenCV](https://opencv.org)** — Frame differencing & motion analysis
- **[dlib](http://dlib.net)** — Underlying face detection (dependency of face_recognition)

---

## 🔮 Future Improvements

- [ ] Real microphone input for actual noise level measurement
- [ ] Email / SMS notifications via Twilio or SendGrid
- [ ] Multi-classroom support
- [ ] Attendance tracking as a secondary feature
- [ ] Mobile app companion for teacher alerts
- [ ] Replace `face_recognition` with `InsightFace` for faster cloud deploys

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change.

---

## 📄 License

[MIT](LICENSE)

---

*Built for Ideathon 26-27 — AI-Powered Automated Teacher Presence Monitoring*
