# Face Rater

A computer vision app that analyzes facial geometry from photos and produces a structured attractiveness and symmetry report.

Built with **Python**, **OpenCV**, and **MediaPipe FaceMesh**.

## Features

- Dual-view analysis: upload a **front** photo and a **side profile** photo
- Scores jaw width, cheekbone prominence, facial symmetry, eye area, chin and jaw projection
- Gender-specific sexual dimorphism calculation
- Tkinter GUI with file picker — no command-line required
- Soft scoring calibrated to realistic human proportions

## Usage

```bash
pip install opencv-python mediapipe
python face_rater.py
```

1. Select gender (Male / Female)
2. Upload a front-facing photo
3. Upload a side profile photo
4. Click **Run Analysis**

## Version History

| Version | File | Description |
|---|---|---|
| v0 precursor | `versions/v0_precursor_jaw_width.py` | Early jaw/face-width geometry study |
| v1.00 | `versions/v1_00_static.py` | Static single-image CLI rater |
| v1.11 | `versions/v1_11.py` | First full CLI with detailed report (jaw, symmetry, cheeks, skin, FWHR) |
| v1.12 – v1.21 | `versions/v1_12.py` … | Tkinter GUI iterations — progressive UX refinement |
| v2.00 | `versions/v2_00.py` | Dual-view (front + side) introduced; redesigned scoring |
| **v2.01** | `face_rater.py` ← **main** | Fixed depth normalization, anatomically realistic projections |
| v2.02b | `versions/v2_02b_scoring.py` | Soft-scoring algorithm improvement (no GUI — logic reference) |

## Stack

`Python` · `OpenCV` · `MediaPipe` · `Tkinter` · `math`
