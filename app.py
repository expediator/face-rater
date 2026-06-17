import streamlit as st
import cv2
import mediapipe as mp
import math
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Rater · expediator", page_icon="🎭", layout="centered")

st.markdown("""
<style>
  .block-container { max-width: 720px; }
</style>
""", unsafe_allow_html=True)

st.title("🎭 Face Rater v2.01")
st.markdown(
    "Facial geometry analysis using **MediaPipe FaceMesh**. "
    "Upload a front photo and a side profile to get a structured report.  \n"
    "[GitHub](https://github.com/expediator/face-rater) · "
    "[Portfolio](https://expediator.github.io/resume/)"
)
st.divider()

# ── Analysis logic ──────────────────────────────────────────────
mp_face   = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def clamp(x, lo=1, hi=10):
    return max(lo, min(hi, x))

def score_from_ideal(value, lo, hi):
    if lo <= value <= hi:
        return 9.0
    return clamp(9.0 - min(abs(value - lo), abs(value - hi)) * 120)

def analyze_front(bgr):
    res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError("No face detected in FRONT photo. Use a well-lit, straight-on image.")
    lm = res.multi_face_landmarks[0].landmark
    jaw_w   = dist(lm[234], lm[454])
    face_h  = dist(lm[10],  lm[152])
    cheek_w = dist(lm[50],  lm[280])
    eye_l   = dist(lm[33],  lm[133])
    eye_r   = dist(lm[362], lm[263])
    return {
        "jaw":      score_from_ideal(jaw_w / face_h,   0.75, 0.85),
        "cheek":    score_from_ideal(cheek_w / jaw_w,  1.1,  1.3),
        "symmetry": clamp(10 - abs(eye_l - eye_r) * 250),
        "eye":      clamp(((eye_l + eye_r) / 2) * 18),
        "face_h":   face_h,
    }

def analyze_side(bgr, face_h):
    res = face_mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError("No face detected in SIDE photo. Use a clear profile image.")
    lm = res.multi_face_landmarks[0].landmark
    nose, chin, jaw = lm[1], lm[152], lm[234]
    return {
        "chin":     score_from_ideal(abs(chin.z - nose.z) / face_h, 0.02, 0.045),
        "jaw_proj": score_from_ideal(abs(jaw.z  - nose.z) / face_h, 0.02, 0.04),
    }

def full_analysis(front_bgr, side_bgr, gender):
    f = analyze_front(front_bgr)
    s = analyze_side(side_bgr, f["face_h"])
    harmony    = clamp((f["jaw"] + f["cheek"] + f["symmetry"] + s["chin"] + s["jaw_proj"]) / 5)
    dimorphism = clamp((f["jaw"] + s["jaw_proj"]) / 2) if gender == "Male" else clamp((f["eye"] + f["cheek"]) / 2)
    face_rating    = clamp(harmony * 0.65 + f["symmetry"] * 0.35)
    attractiveness = clamp((face_rating + f["eye"]) / 2)
    return dict(front=f, side=s, harmony=harmony, dimorphism=dimorphism,
                face_rating=face_rating, attractiveness=attractiveness,
                potential=clamp(attractiveness + 2))

def bar(v):
    return "█" * round(v) + "░" * (10 - round(v)) + f"  {v:.1f}/10"

def label(v):
    if v >= 8.5: return "🔥 Excellent"
    if v >= 7.5: return "✅ Above Average"
    if v >= 6.0: return "📊 Average"
    if v >= 5.0: return "📉 Below Average"
    return "⚠️ Needs Work"

# ── UI ──────────────────────────────────────────────────────────
gender = st.radio("Select gender", ["Male", "Female"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    front_file = st.file_uploader("📸 Front photo", type=["jpg","jpeg","png"])
    if front_file:
        st.image(front_file, use_column_width=True)
with col2:
    side_file = st.file_uploader("📸 Side profile photo", type=["jpg","jpeg","png"])
    if side_file:
        st.image(side_file, use_column_width=True)

st.divider()

if st.button("▶ Run Analysis", type="primary", disabled=not (front_file and side_file)):
    with st.spinner("Analysing facial geometry…"):
        try:
            def to_bgr(f):
                return cv2.cvtColor(np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)

            r = full_analysis(to_bgr(front_file), to_bgr(side_file), gender)
            f, s = r["front"], r["side"]

            st.success("Analysis complete!")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Front View**")
                st.code(
                    f"Jaw Width       {bar(f['jaw'])}\n"
                    f"Cheekbones      {bar(f['cheek'])}\n"
                    f"Symmetry        {bar(f['symmetry'])}\n"
                    f"Eye Area        {bar(f['eye'])}", language=None)
            with c2:
                st.markdown("**Side Profile**")
                st.code(
                    f"Chin Projection {bar(s['chin'])}\n"
                    f"Jaw Projection  {bar(s['jaw_proj'])}", language=None)

            st.markdown("**Aggregate**")
            st.code(
                f"Facial Harmony      {bar(r['harmony'])}\n"
                f"Sexual Dimorphism   {bar(r['dimorphism'])}\n"
                f"Face Rating         {bar(r['face_rating'])}\n"
                f"Attractiveness      {bar(r['attractiveness'])}\n"
                f"Max Potential       {bar(r['potential'])}", language=None)

            st.markdown(f"### Overall: {label(r['attractiveness'])}")

        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error: {e}")

st.divider()
st.caption("Scores are based on geometric proportions only — not a medical or scientific assessment.")
