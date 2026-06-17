import cv2
import mediapipe as mp
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

# ================= BASIC UTILS =================
def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def clamp(x, lo=1, hi=10):
    return max(lo, min(hi, x))

def smooth_score(value, ideal, tolerance, weight=6):
    """
    value: measured ratio
    ideal: target value
    tolerance: acceptable human variation
    """
    deviation = abs(value - ideal) / tolerance
    score = 10 - (deviation ** 2) * weight
    return clamp(score)

# ================= FRONT VIEW =================
def analyze_front(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        raise ValueError("No face detected in FRONT image")

    lm = res.multi_face_landmarks[0].landmark

    face_h = dist(lm[10], lm[152])
    face_w = dist(lm[234], lm[454])
    cheek_w = dist(lm[50], lm[280])
    eye_l = dist(lm[33], lm[133])
    eye_r = dist(lm[362], lm[263])

    # Ratios
    jaw_ratio = face_w / face_h
    cheek_ratio = cheek_w / face_w
    eye_ratio = ((eye_l + eye_r) / 2) / face_w

    # Scores (SOFTLY SCALED)
    jaw_score = smooth_score(jaw_ratio, 0.80, 0.12)
    cheek_score = smooth_score(cheek_ratio, 1.18, 0.15)
    eye_score = smooth_score(eye_ratio, 0.19, 0.04)

    symmetry = clamp(10 - abs(eye_l - eye_r) * 180)

    return {
        "jaw": jaw_score,
        "cheek": cheek_score,
        "eye": eye_score,
        "symmetry": symmetry,
        "face_h": face_h
    }

# ================= SIDE VIEW =================
def analyze_side(img, face_h):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        raise ValueError("No face detected in SIDE image")

    lm = res.multi_face_landmarks[0].landmark
    nose = lm[1]
    chin = lm[152]
    jaw = lm[234]

    chin_proj = abs(chin.z - nose.z) / face_h
    jaw_proj = abs(jaw.z - nose.z) / face_h

    chin_score = smooth_score(chin_proj, 0.035, 0.02)
    jaw_proj_score = smooth_score(jaw_proj, 0.030, 0.02)

    return {
        "chin": chin_score,
        "jaw_proj": jaw_proj_score
    }

# ================= FINAL AGGREGATION =================
def analyze(front_img, side_img, gender):
    f = analyze_front(front_img)
    s = analyze_side(side_img, f["face_h"])

    # Harmony favors balance, not punishment
    harmony = clamp(
        (f["jaw"] + f["cheek"] + f["eye"] +
         s["chin"] + s["jaw_proj"] + f["symmetry"]) / 6
    )

    if gender == "Male":
        dimorphism = clamp((f["jaw"] + s["jaw_proj"] + s["chin"]) / 3 + 0.5)
    else:
        dimorphism = clamp((f["eye"] + f["cheek"] + f["symmetry"]) / 3 + 0.5)

    face_rating = clamp(harmony * 0.7 + f["symmetry"] * 0.3)
    attractiveness = clamp((face_rating + f["eye"] + dimorphism) / 3)
    potential = clamp(attractiveness + 1.5)

    report = f"""
================ FACE RATER v2.02 (CALIBRATED) =================

Gender: {gender}

--- FRONT VIEW ---
Jaw Width: {f['jaw']:.1f}/10
Cheekbones: {f['cheek']:.1f}/10
Eye Area: {f['eye']:.1f}/10
Facial Symmetry: {f['symmetry']:.1f}/10

--- SIDE PROFILE ---
Chin Projection: {s['chin']:.1f}/10
Jaw Projection: {s['jaw_proj']:.1f}/10

--- OVERALL ---
Facial Harmony: {harmony:.1f}/10
Sexual Dimorphism: {dimorphism:.1f}/10
Face Rating (Structure): {face_rating:.1f}/10
Attractiveness: {attractiveness:.1f}/10
Maximum Potential: {potential:.1f}/10

--- CALIBRATION NOTES ---
• Near-ideal features remain HIGH
• No harsh penalties
• PSL 6–7 ≈ 6.5–7.5 output
• Harmony boosts instead of punishing

============================================================
"""
    return report

# ================= GUI =================
class App:
    def __init__(self, root):
        root.title("Face Rater v2.02 — Calibrated")
        root.geometry("950x720")

        self.gender = tk.StringVar(value="Male")
        self.front = None
        self.side = None

        ttk.Label(root, text="Face Rater v2.02 (Calibrated)", font=("Arial", 20)).pack(pady=10)

        g = ttk.Frame(root)
        g.pack()
        ttk.Radiobutton(g, text="Male", variable=self.gender, value="Male").pack(side="left", padx=10)
        ttk.Radiobutton(g, text="Female", variable=self.gender, value="Female").pack(side="left", padx=10)

        ttk.Button(root, text="Upload FRONT Photo", command=self.load_front).pack(pady=5)
        ttk.Button(root, text="Upload SIDE Photo", command=self.load_side).pack(pady=5)
        ttk.Button(root, text="Run Analysis", command=self.run).pack(pady=10)

        self.text = tk.Text(root, width=110, height=32, wrap="word")
        self.text.pack()

    def load_front(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        if path:
            self.front = cv2.imread(path)

    def load_side(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        if path:
            self.side = cv2.imread(path)

    def run(self):
        if self.front is None or self.side is None:
            messagebox.showerror("Error", "Upload both FRONT and SIDE images")
            return
        result = analyze(self.front, self.side, self.gender.get())
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, result)

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
