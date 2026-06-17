# 🎭 FaceMetrics — AI Facial Geometry Analysis

Browser-based facial geometry analyzer using **MediaPipe FaceMesh**. Upload photos or use your live webcam. Scores jaw width, cheekbones, symmetry, eye area, chin and jaw projection — computed entirely client-side, no server required.

**Live:** [expediator.github.io/face-rater](https://expediator.github.io/face-rater/) &nbsp;·&nbsp; **Portfolio:** [expediator.github.io/resume](https://expediator.github.io/resume/)

---

## How It Works

1. Select a gender (affects dimorphism scoring)
2. Provide a **front-facing photo** + **side-profile photo** (upload file OR capture from webcam)
3. MediaPipe FaceMesh detects 468 landmarks on each face
4. Ratios between landmark positions are scored against ideal ranges
5. 11 individual scores displayed on color-coded bars (scale 1–10)

## Scores

| Score | What's Measured | Ideal Range |
|---|---|---|
| Jaw Width | Jaw width ÷ face height | 0.75 – 0.85 |
| Cheekbones | Cheek width ÷ jaw width | 1.1 – 1.3 |
| Symmetry | Left vs right eye width difference | smaller = better |
| Eye Area | Average eye opening (normalized) | larger = better |
| Chin Projection | Chin depth vs nose (side view, z-axis) | 0.02 – 0.045 |
| Jaw Projection | Jaw depth vs nose (side view, z-axis) | 0.02 – 0.04 |

Aggregate: Facial Harmony, Dimorphism, Face Rating, Attractiveness, Max Potential.

## Features

- 📁 **File upload** or **📷 webcam capture** for each photo zone
- ⚡ **Auto brightness enhancement** — dark/poorly lit photos boosted via Canvas API before analysis
- 🚫 **No server** — 100% browser-based, MediaPipe JS runs locally
- 🌐 **No install** — works in any modern browser

## Version History (16 versions)

| Version | What changed |
|---|---|
| v0 | Single image, raw ratio output in terminal |
| v1.x | Scoring scale (1–10), basic GUI |
| v2.0 | Dual-photo (front + side), Tkinter desktop GUI |
| v2.01 | Rewrote as browser JS app using MediaPipe JS CDN |
| v2.01+ | Added webcam capture, brightness filter, renamed FaceMetrics |

## Tech Stack

- **MediaPipe FaceMesh 0.4** — 468-point facial landmark detection (JS, via CDN)
- **Canvas API** — image preprocessing and brightness normalization
- **Vanilla JS / HTML / CSS** — no frameworks, no build step
- **GitHub Pages** — free static hosting from `/docs` folder

## Files

```
face-rater/
├── docs/
│   └── index.html       ← Web app (deployed to GitHub Pages)
├── face_rater.py        ← Python desktop app (v2.01, Tkinter + OpenCV)
├── app.py               ← Deprecated Streamlit version (ignore)
├── requirements.txt     ← Python deps for desktop app
├── versions/            ← Full history v0 → v2.02b (16 files)
└── README.md
```

## Run the Desktop App (Python)

```bash
pip install mediapipe opencv-python
python face_rater.py
```

## Disclaimer

Scores are geometric proportions only — not medical, scientific, or beauty judgements. Educational/portfolio demo only.
