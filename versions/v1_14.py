import cv2
import mediapipe as mp
import numpy as np
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox

# ====================== UTILITIES ======================
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def clamp(x, a=1, b=10):
    return max(a, min(b, x))

def meaning(s):
    if s >= 8: return "Excellent"
    if s >= 6.5: return "Good"
    if s >= 5: return "Average"
    return "Below Average"

# ====================== FILE PICKER (FIXED) ======================
def pick_image():
    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")]
        )
    except Exception as e:
        messagebox.showerror("Error", str(e))
        path = ""
    root.destroy()
    return path

# ====================== MAIN ANALYSIS ======================
def analyze(image, is_male):
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(static_image_mode=True)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, "No face detected."

    face = results.multi_face_landmarks[0]

    # Landmarks
    jaw_l, jaw_r = face.landmark[234], face.landmark[454]
    chin, forehead = face.landmark[152], face.landmark[10]
    cheek_l, cheek_r = face.landmark[93], face.landmark[323]
    eye_l_o, eye_l_i = face.landmark[33], face.landmark[133]
    eye_r_o, eye_r_i = face.landmark[263], face.landmark[362]

    # Measurements
    jaw_ratio = distance(jaw_l, jaw_r) / distance(forehead, chin)
    fwhr = distance(cheek_l, cheek_r) / distance(forehead, chin)
    eye_width = distance(eye_l_i, eye_l_o)
    canthal = (eye_r_o.y - eye_l_o.y)

    # Scores
    jaw = clamp(jaw_ratio * 12)
    fwhr_s = clamp(fwhr * 5)
    eye = clamp(eye_width * 40)
    symmetry = clamp((1 - abs(eye_l_o.y - eye_r_o.y)) * 10)
    canthal_s = clamp(canthal * 25 + 5)

    normal = clamp((jaw + fwhr_s + eye + symmetry) / 4)
    psl = clamp(normal - 0.5)

    if is_male:
        gender_score = clamp((jaw * 1.4 + fwhr_s + canthal_s) / 3)
        gender_label = "Masculinity"
    else:
        gender_score = clamp((eye * 1.4 + symmetry + (10 - jaw)) / 3)
        gender_label = "Femininity"

    report = f"""
================ FACE ANALYSIS REPORT ================

Jawline      : {jaw:.1f}/10 ({meaning(jaw)})
Symmetry     : {symmetry:.1f}/10 ({meaning(symmetry)})
Eye Area     : {eye:.1f}/10 ({meaning(eye)})
Canthal Tilt : {canthal_s:.1f}/10
FWHR         : {fwhr_s:.1f}/10

{gender_label}: {gender_score:.1f}/10 ({meaning(gender_score)})

Normal Scale : {normal:.1f}/10
PSL Scale    : {psl:.1f}/10

Positives:
"""

    if jaw > 7: report += "- Strong jaw structure\n"
    if eye > 7: report += "- Attractive eye area\n"
    if symmetry > 7: report += "- Good symmetry\n"
    if gender_score > 7: report += f"- High {gender_label.lower()}\n"

    report += "\nNegatives:\n"
    if jaw < 5: report += "- Weak jaw definition\n"
    if eye < 5: report += "- Eye area could improve\n"
    if symmetry < 5: report += "- Facial asymmetry present\n"

    report += "\n===================================================="

    return report, None

# ====================== GUI ======================
def start_analysis():
    global sex_var, output

    path = pick_image()
    if not path:
        return

    image = cv2.imread(path)
    if image is None:
        messagebox.showerror("Error", "Unable to load image.")
        return

    is_male = sex_var.get() == 1
    report, err = analyze(image, is_male)

    if err:
        output.delete("1.0", tk.END)
        output.insert(tk.END, err)
    else:
        output.delete("1.0", tk.END)
        output.insert(tk.END, report)

# ====================== APP WINDOW ======================
app = tk.Tk()
app.title("Face Rater (Personal Use)")
app.geometry("700x600")

tk.Label(app, text="Select Biological Sex").pack()

sex_var = tk.IntVar(value=1)
tk.Radiobutton(app, text="Male", variable=sex_var, value=1).pack()
tk.Radiobutton(app, text="Female", variable=sex_var, value=2).pack()

tk.Button(app, text="Upload Image & Analyze", command=start_analysis).pack(pady=10)

output = tk.Text(app, wrap=tk.WORD)
output.pack(expand=True, fill=tk.BOTH)

app.mainloop()
