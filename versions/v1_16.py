import cv2
import mediapipe as mp
import math
import tkinter as tk
from tkinter import filedialog, messagebox

# ===================== UTILITIES =====================
def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def clamp(x, lo=1, hi=10):
    return max(lo, min(hi, x))

def meaning(s):
    if s >= 8: return "Excellent"
    if s >= 6.5: return "Good"
    if s >= 5: return "Average"
    return "Below Average"

# ===================== FILE PICKER =====================
def pick_image():
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
    )
    root.destroy()
    return path

# ===================== FACE ANALYSIS =====================
def analyze_face(image, is_male):
    mp_face = mp.solutions.face_mesh
    mesh = mp_face.FaceMesh(static_image_mode=True)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)

    if not res.multi_face_landmarks:
        return None, "No face detected."

    f = res.multi_face_landmarks[0]

    # ---------- LANDMARKS ----------
    jaw_l, jaw_r = f.landmark[234], f.landmark[454]
    chin, forehead = f.landmark[152], f.landmark[10]
    cheek_l, cheek_r = f.landmark[93], f.landmark[323]
    eye_lo, eye_li = f.landmark[33], f.landmark[133]
    eye_ro, eye_ri = f.landmark[263], f.landmark[362]

    # ---------- MEASUREMENTS ----------
    face_h = dist(forehead, chin)
    jaw_w = dist(jaw_l, jaw_r)
    face_w = dist(cheek_l, cheek_r)
    eye_w = dist(eye_lo, eye_li)

    jaw_ratio = jaw_w / face_h
    fwhr = face_w / face_h
    eye_ratio = eye_w / face_w
    canthal_raw = (eye_ro.y - eye_lo.y)

    # ---------- BASE SCORES ----------
    jaw = clamp(jaw_ratio * 12)
    fwhr_s = clamp(fwhr * 5)
    eye = clamp(eye_ratio * 80)
    symmetry = clamp((1 - abs(eye_lo.y - eye_ro.y)) * 10)
    canthal = clamp(canthal_raw * 25 + 5)

    # ---------- FACIAL HARMONY (STRUCTURAL) ----------
    ideal_fwhr = 1.9
    harmony = clamp(10 - abs(fwhr - ideal_fwhr) * 6)

    # ---------- FACE RATING (OBJECTIVE) ----------
    face_rating = clamp(
        jaw*0.25 +
        symmetry*0.25 +
        harmony*0.3 +
        eye*0.2
    )

    # ---------- GENDER TRAITS ----------
    if is_male:
        dimorphism = clamp((jaw*1.4 + fwhr_s + canthal) / 3)
        dimorphism_label = "Masculinity"
    else:
        dimorphism = clamp((eye*1.3 + symmetry + (10 - jaw)) / 3)
        dimorphism_label = "Femininity"

    # ---------- ATTRACTIVENESS ----------
    attractiveness = clamp(
        face_rating*0.55 +
        dimorphism*0.3 +
        eye*0.1 +
        canthal*0.05
    )

    # ---------- REPORT ----------
    report = f"""
================ PHOTO RATER 1.16 =================

STRUCTURAL SCORES
-------------------------------------
Jawline          : {jaw:.1f}/10 ({meaning(jaw)})
Symmetry         : {symmetry:.1f}/10 ({meaning(symmetry)})
Eye Proportion   : {eye:.1f}/10 ({meaning(eye)})
FWHR             : {fwhr_s:.1f}/10
Facial Harmony   : {harmony:.1f}/10 ({meaning(harmony)})

FACE RATING (OBJECTIVE)
>>> {face_rating:.1f}/10 <<<

-------------------------------------

SEXUAL DIMORPHISM
{dimorphism_label}     : {dimorphism:.1f}/10 ({meaning(dimorphism)})

ATTRACTIVENESS (PERCEIVED)
>>> {attractiveness:.1f}/10 <<<

-------------------------------------

INTERPRETATION
"""
    if face_rating >= 7:
        report += "- Strong facial structure and balance\n"
    if attractiveness >= 7:
        report += "- Likely perceived as attractive by most observers\n"
    if dimorphism < 5:
        report += f"- {dimorphism_label} traits are mild\n"

    report += "\n=================================================="

    return report, None

# ===================== GUI =====================
def start():
    path = pick_image()
    if not path:
        return

    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "Could not load image.")
        return

    is_male = sex_var.get() == 1
    report, err = analyze_face(img, is_male)

    output.delete("1.0", tk.END)
    output.insert(tk.END, err if err else report)

app = tk.Tk()
app.title("Photo Rater 1.16 (Personal Use)")
app.geometry("780x700")

tk.Label(app, text="Select Biological Sex").pack()
sex_var = tk.IntVar(value=1)
tk.Radiobutton(app, text="Male", variable=sex_var, value=1).pack()
tk.Radiobutton(app, text="Female", variable=sex_var, value=2).pack()

tk.Button(app, text="Upload Photo & Analyze", command=start).pack(pady=10)

output = tk.Text(app, wrap=tk.WORD)
output.pack(expand=True, fill=tk.BOTH)

app.mainloop()
