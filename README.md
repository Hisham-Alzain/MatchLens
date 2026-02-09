# ⚽ MatchLens
**Turning football matches into data you can actually use.**

---

## Quick Pitch
**MatchLens** is a football analytics prototype developed by **TupCode**.  
It transforms match video into accurate, pitch‑level tracking and team data — built for analysts, coaches, and football enthusiasts who want to understand what really happens between kickoff and the final whistle, not just the scoreline.

---

## What's Working Right Now
The current version focuses on building a **rock‑solid vision and tracking foundation**.  
The following features are implemented and stable:

- **Detection** — Frame‑level detection of players and ball using modern object detection models.  
- **Homography** — Perspective transformation from broadcast video to real‑world pitch coordinates (Demo version).  
- **Tracking** — Multi‑object tracking that maintains consistent player identities over time.  
- **Team Assignment** — Automatic team labeling via jersey color clustering.

---

## Why MatchLens?
Football is a game of **space, movement, and timing** — and MatchLens is built to capture exactly that.  
By converting raw video into **metric pitch data**, the system enables:

- Distance and movement analysis  
- Formation and positioning visualization  
- Tactical insights and spatial metrics  

This tracking layer lays the foundation for **passes, possession, and advanced analytics** in future updates.

---

## How It Works (High-Level)
1. **Video Ingestion** — Decode match footage into frames.  
2. **Detection** — Identify players and ball in each frame.  
3. **Tracking** — Link detections across frames to form trajectories.  
4. **Homography** — Map image coordinates onto a standardized football pitch.  
5. **Team Assignment** — Group players by team based on visual appearance.  
6. **Analysis‑Ready Output** — Generate clean trajectory data for future event detection.

---

## Installation

### Requirements
- Python 3.10+  
- FFmpeg  
- CUDA‑enabled GPU (recommended for performance)

### Setup
```bash
git clone https://github.com/your-org/tactixai.git
cd MatchLens
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Roadmap
Planned features for future releases:

⚽ Football event detection (passes, shots, tackles, etc.)

🔄 Possession segmentation and automatic highlight clips

📊 Tactical analytics (xG, xT, pitch control, heatmaps)

🧠 Player and team style analysis

🌟 Automatic detection of standout player performances

## About TupCode
TupCode is a software and AI startup focused on building practical AI systems for sports analytics and intelligent software products.

🌐 Website: https://tupcode.com

📧 Contact: info@tupcode.com
