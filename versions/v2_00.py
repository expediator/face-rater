import cv2
import mediapipe as mp
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ================= MEDIAPIPE =================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

# ================= HELPERS =================
def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def clamp(x, lo=1, hi=10):
    return max(lo, min(hi, x))

def score_range(v, lo, hi):
    if lo <= v <= hi:
        return 9
    return clamp(9 - abs(v - (lo + hi) / 2) * 10)

# ================= FRONT ANALYSIS =================
def analyze_front(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        raise ValueError("No face detected in FRONT photo")

    lm = res.multi_face_landmarks[0].landmark

    jaw_w = dist(lm[234], lm[454])
    face_h = dist(lm[10], lm[152])
    cheek_w = dist(lm[50], lm[280])
    eye_l = dist(lm[33], lm[133])
    eye_r = dist(lm[362], lm[263])

    fwhr = jaw_w / face_h
    cheek_jaw = cheek_w / jaw_w
    symmetry = 10 - abs(eye_l - eye_r) * 250
    eye_score = clamp(((eye_l + eye_r) / 2) * 18)

    jaw_score = score_range(fwhr, 0.75, 0.85)
    cheek_score = score_range(cheek_jaw, 1.1, 1.3)
    symmetry_score = clamp(symmetry)

    return {
        "jaw": jaw_score,
        "cheek": cheek_score,
        "symmetry": symmetry_score,
        "eye": eye_score
    }

# ================= SIDE ANALYSIS =================
def analyze_side(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        raise ValueError("No face detected in SIDE photo")

    lm = res.multi_face_landmarks[0].landmark

    chin = lm[152]
    nose = lm[1]
    forehead = lm[10]
    neck = lm[234]

    chin_proj = abs(chin.z - nose.z)
    jaw_proj = abs(chin.z - forehead.z)
    nose_proj = abs(nose.z - forehead.z)

    chin_score = clamp(chin_proj * 60)
    jaw_proj_score = clamp(jaw_proj * 60)
    nose_score = clamp(10 - nose_proj * 40)

    return {
        "chin": chin_score,
        "jaw_proj": jaw_proj_score,
        "nose": nose_score
    }

# ================= FULL ANALYSIS =================
def analyze(front_img, side_img, gender):
    front = analyze_front(front_img)
    side = analyze_side(side_img)

    harmony = clamp(
        (front["jaw"] + front["cheek"] + front["symmetry"] +
         side["chin"] + side["jaw_proj"]) / 5
    )

    if gender == "Male":
        dimorphism = clamp((front["jaw"] + side["jaw_proj"]) / 2)
    else:
        dimorphism = clamp((front["eye"] + front["cheek"]) / 2)

    face_rating = clamp(harmony * 0.6 + front["symmetry"] * 0.4)
    attractiveness = clamp((face_rating + front["eye"]) / 2)
    potential = clamp(attractiveness + 2)

    report = f"""
================ FACE RATER v2.00 (DUAL VIEW) =================

Gender: {gender}

--- FRONT VIEW ---
Jaw Width: {front['jaw']:.1f}/10
Cheekbones: {front['cheek']:.1f}/10
Symmetry: {front['symmetry']:.1f}/10
Eye Area: {front['eye']:.1f}/10

--- SIDE PROFILE ---
Chin Projection: {side['chin']:.1f}/10
Jaw Projection: {side['jaw_proj']:.1f}/10
Nose Balance: {side['nose']:.1f}/10

--- AGGREGATES ---
Facial Harmony: {harmony:.1f}/10
Sexual Dimorphism: {dimorphism:.1f}/10
Face Rating (Structure): {face_rating:.1f}/10
Attractiveness: {attractiveness:.1f}/10
Maximum Potential: {potential:.1f}/10

--- NOTES ---
• Front view judges balance & symmetry
• Side view judges projection & profile
• Dual-view improves accuracy significantly
• Potential assumes grooming + fat loss + styling

===============================================================
"""
    return report

# ================= GUI =================
class App:
    def __init__(self, root):
        root.title("Face Rater v2.00 — Dual View")
        root.geometry("1000x750")

        self.gender = tk.StringVar(value="Male")
        self.front_img = None
        self.side_img = None

        ttk.Label(root, text="Face Rater v2.00 (Front + Side)", font=("Arial", 20)).pack(pady=10)

        g = ttk.Frame(root)
        g.pack()
        ttk.Radiobutton(g, text="Male", variable=self.gender, value="Male").pack(side="left", padx=10)
        ttk.Radiobutton(g, text="Female", variable=self.gender, value="Female").pack(side="left", padx=10)

        ttk.Button(root, text="Upload FRONT Photo", command=self.load_front).pack(pady=5)
        ttk.Button(root, text="Upload SIDE Photo", command=self.load_side).pack(pady=5)
        ttk.Button(root, text="Run Analysis", command=self.run).pack(pady=10)

        self.text = tk.Text(root, width=120, height=30, wrap="word")
        self.text.pack(pady=10)

    def load_front(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        if path:
            self.front_img = cv2.imread(path)

    def load_side(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        if path:
            self.side_img = cv2.imread(path)

    def run(self):
        if self.front_img is None or self.side_img is None:
            messagebox.showerror("Error", "Please upload BOTH front and side photos")
            return
        report = analyze(self.front_img, self.side_img, self.gender.get())
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, report)

# ================= RUN =================
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
