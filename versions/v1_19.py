"""
PHOTO RATER 2.1 – FINAL EXTENDED
"""

import cv2, mediapipe as mp, numpy as np, math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime

# ===================== UTILS =====================
def clamp(v): return max(1, min(10, float(v)))
def mean(v): return sum(v)/len(v) if v else 0

# ===================== MEDIAPIPE =====================
mp_face = mp.solutions.face_mesh
mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

# ===================== CORE ANALYSIS =====================
def analyze_face(img, is_male):
    res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None

    L = res.multi_face_landmarks[0].landmark
    d = lambda a,b: math.dist((L[a].x,L[a].y),(L[b].x,L[b].y))

    # --- Geometry ---
    face_h = d(10,152)
    face_w = d(234,454)
    jaw_w  = d(234,454)
    cheek = d(127,356)
    eye_d = d(33,263)
    mid = d(9,2)
    low = d(2,152)
    nose = d(98,327)
    mouth = d(61,291)

    # --- Scores ---
    symmetry = clamp(10 - abs(L[33].y - L[263].y) * 40)
    thirds = clamp(10 - abs(mid/low - 1.05) * 10)
    jaw = clamp((jaw_w/face_h)*12)
    cheekbones = clamp((cheek/face_w)*14)
    eye_spacing = clamp(10 - abs((eye_d/face_w) - 0.46) * 25)
    nose_balance = clamp(10 - abs((nose/mouth) - 0.5) * 20)

    harmony = clamp(mean([
        symmetry, thirds, jaw,
        cheekbones, eye_spacing, nose_balance
    ]))

    # --- Skin ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).std()
    skin = clamp(10 - texture/7)

    # --- Eye Shape ---
    eyelid = abs(L[159].y - L[145].y)
    eye_shape = clamp(5 + eyelid * 35)

    # --- Face Rating (Objective) ---
    face_rating = clamp(mean([
        harmony*1.3,
        symmetry,
        jaw,
        cheekbones,
        thirds
    ]))

    # --- Sexual Dimorphism ---
    if is_male:
        dim = clamp(mean([jaw*1.2, cheekbones, harmony]))
        dim_label = "Masculinity"
    else:
        dim = clamp(mean([eye_shape*1.4, symmetry, harmony]))
        dim_label = "Femininity"

    # --- Attractiveness (Perceived) ---
    attractiveness = clamp(mean([
        face_rating*1.4,
        dim,
        eye_shape,
        skin
    ]))

    # --- Potential (MAX achievable) ---
    improvables = clamp(mean([skin, eye_shape]))
    potential = clamp(face_rating + (10 - improvables)*0.6)

    # --- Age Estimation (rough) ---
    age = int(18 + (texture * 0.8))
    age = max(16, min(age, 65))

    # --- Reliability ---
    angle_penalty = abs(L[234].x - 0.15)
    confidence = "High" if angle_penalty < 0.04 else "Medium"
    if angle_penalty > 0.08:
        confidence = "Low"

    return {
        "Face Rating": round(face_rating,2),
        "Attractiveness": round(attractiveness,2),
        "Potential (Max)": round(potential,2),
        dim_label: round(dim,2),
        "Facial Harmony": round(harmony,2),
        "Jaw": round(jaw,2),
        "Cheekbones": round(cheekbones,2),
        "Symmetry": round(symmetry,2),
        "Eye Shape": round(eye_shape,2),
        "Skin Quality": round(skin,2),
        "Estimated Age": f"{age} ± 3",
        "Reliability": confidence,
        "Image": img
    }

# ===================== GUI =====================
class App:
    def __init__(self,root):
        root.title("Photo Rater 1.19")
        self.img=None
        self.rep=None

        self.sex=tk.StringVar(value="M")

        tk.Radiobutton(root,text="Male",variable=self.sex,value="M").pack()
        tk.Radiobutton(root,text="Female",variable=self.sex,value="F").pack()
        tk.Button(root,text="Upload Image",command=self.load).pack()
        tk.Button(root,text="Analyze",command=self.run).pack()

        self.preview=tk.Label(root,bg="#222",width=550,height=350)
        self.preview.pack(pady=6)

        self.out=tk.Text(root,width=80,height=22)
        self.out.pack()

    def load(self):
        p=filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        if not p: return
        self.img=cv2.imread(p)
        im=Image.fromarray(cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB))
        im.thumbnail((550,350))
        self.tk=ImageTk.PhotoImage(im)
        self.preview.configure(image=self.tk)

    def run(self):
        if self.img is None: return
        self.rep=analyze_face(self.img,self.sex.get()=="M")
        if not self.rep:
            messagebox.showerror("Error","No face detected")
            return
        self.out.delete(1.0,tk.END)
        for k,v in self.rep.items():
            if k!="Image":
                self.out.insert(tk.END,f"{k}: {v}\n")

# ===================== RUN =====================
if __name__=="__main__":
    root=tk.Tk()
    App(root)
    root.mainloop()
