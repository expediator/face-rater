<div align="center">

# 🎭 FaceMetrics

**AI facial geometry analysis — 468 landmarks, 100% in-browser**

[![Live](https://img.shields.io/badge/Try_It-Live-3b9eff?style=flat-square&logo=github)](https://expediator.github.io/face-rater/)
[![GitHub](https://img.shields.io/badge/GitHub-expediator%2Fface--rater-181717?style=flat-square&logo=github)](https://github.com/expediator/face-rater)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![MediaPipe](https://img.shields.io/badge/MediaPipe_FaceMesh-0097A7?style=flat-square)](https://mediapipe.dev)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

Upload a front + side photo (or webcam capture) and get instant geometric analysis of your facial proportions. Entirely client-side — no server, no data uploaded.

</div>

---

## Try It

**[expediator.github.io/face-rater](https://expediator.github.io/face-rater/)** — any modern browser, no install.

---

## How It Works

1. Select gender (affects dimorphism scoring)
2. Provide **front photo** + **side photo** (file upload or webcam)
3. MediaPipe FaceMesh detects 468 3D landmarks per face
4. Ratios scored against ideal ranges
5. 11 scores displayed on color-coded bars (1–10 scale)

## Scores

| Score | What's Measured |
|-------|----------------|
| Jaw Width | Jaw width ÷ face height (ideal: 0.75–0.85) |
| Cheekbones | Cheek width ÷ jaw width (ideal: 1.1–1.3) |
| Symmetry | Left vs right eye width difference |
| Eye Area | Average eye opening normalized |
| Chin Projection | Chin depth vs nose (side view) |
| Jaw Projection | Jaw depth vs nose (side view) |
| + 5 aggregates | Facial Harmony · Dimorphism · Face Rating · Attractiveness · Max Potential |

## Features

- File upload or live webcam capture for each photo zone
- Auto brightness enhancement for poorly-lit photos (Canvas API)
- 100% browser-based — MediaPipe runs via CDN, nothing uploaded

## Version History (16 versions)

| Version | Milestone |
|---------|-----------|
| v0 | CLI script, terminal output |
| v1.x | Scoring scale 1–10, basic Tkinter GUI |
| v2.0 | Dual-photo (front + side), full Tkinter GUI |
| v2.01 | Rewrite as browser JS app |
| v2.01+ | Webcam, brightness filter, renamed FaceMetrics |

## Run Python Desktop App

```bash
pip install mediapipe opencv-python
python face_rater.py
```

## Files

```
face-rater/
├── docs/index.html      ← Web app (GitHub Pages)
├── face_rater.py        ← Python desktop app (Tkinter + OpenCV)
├── requirements.txt
├── versions/            ← Full history v0 → v2.02b
└── README.md
```

> **Disclaimer:** Geometric proportions only — not medical or scientific claims.

---

<div align="center">
An <strong>Expediator Inc.</strong> project · © 2026 · <a href="https://expediator.github.io/resume/">Portfolio</a>
</div>