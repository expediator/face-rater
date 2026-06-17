import cv2, mediapipe as mp, math, tkinter as tk
from tkinter import ttk, filedialog, messagebox

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

def dist(a,b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)

def clamp(x): 
    return max(1, min(10, x))

# 🔑 Soft scoring: rewards closeness to ideal
def soft_score(value, ideal, tol_low, tol_high):
    if tol_low <= value <= tol_high:
        return 8.5
    deviation = abs(value - ideal)
    return clamp(8.5 - deviation * 15)

# ================= FRONT =================
def analyze_front(img):
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError("No face detected (front)")
    lm = res.multi_face_landmarks[0].landmark

    face_h = dist(lm[10], lm[152])
    jaw_w  = dist(lm[234], lm[454])
    cheek_w = dist(lm[50], lm[280])
    eye_avg = (dist(lm[33], lm[133]) + dist(lm[362], lm[263])) / 2

    jaw_ratio = jaw_w / face_h
    cheek_ratio = cheek_w / jaw_w
    eye_ratio = eye_avg / face_h

    jaw = soft_score(jaw_ratio, 0.80, 0.74, 0.88)
    cheek = soft_score(cheek_ratio, 1.20, 1.05, 1.35)
    eyes = soft_score(eye_ratio, 0.155, 0.14, 0.18)

    symmetry = clamp(7 + (1 - abs(eye_ratio - 0.155)) * 5)

    return {
        "jaw": jaw,
        "cheek": cheek,
        "eyes": eyes,
        "symmetry": symmetry,
        "face_h": face_h
    }

# ================= SIDE =================
def analyze_side(img, face_h):
    res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError("No face detected (side)")
    lm = res.multi_face_landmarks[0].landmark

    nose = lm[1]
    chin = lm[152]
    jaw  = lm[234]

    chin_proj = abs(chin.z - nose.z) / face_h
    jaw_proj  = abs(jaw.z  - nose.z) / face_h

    chin_score = soft_score(chin_proj, 0.035, 0.02, 0.055)
    jaw_proj_score = soft_score(jaw_proj, 0.03, 0.018, 0.05)

    return chin_score, jaw_proj_score

# ================= FINAL =================
def analyze(front, side, gender):
    f = analyze_front(front)
    chin, jaw_p = analyze_side(side, f["face_h"])

    if gender == "Male":
        dimorphism = clamp((f["jaw"]*0.4 + jaw_p*0.4 + chin*0.2))
    else:
        dimorphism = clamp((f["eyes"]*0.5 + f["cheek"]*0.5))

    harmony = clamp(
        f["jaw"]*0.25 +
        f["cheek"]*0.2 +
        f["eyes"]*0.2 +
        chin*0.15 +
        jaw_p*0.2
    )

    face_rating = clamp((harmony*0.6 + f["symmetry"]*0.4))
    attractiveness = clamp((face_rating + f["eyes"] + dimorphism) / 3)
    potential = clamp(attractiveness + 1.5)

    return f"""
FACE RATER v2.02 (Human-Calibrated)

Jaw Width: {f['jaw']:.1f}
Cheekbones: {f['cheek']:.1f}
Eye Area: {f['eyes']:.1f}
Chin Projection: {chin:.1f}
Jaw Projection: {jaw_p:.1f}

Sexual Dimorphism: {dimorphism:.1f}
Facial Harmony: {harmony:.1f}

FACE STRUCTURE RATING: {face_rating:.1f}
ATTRACTIVENESS: {attractiveness:.1f}
MAX POTENTIAL: {potential:.1f}
"""

# GUI omitted here for brevity (same as before)
