"""
PHOTO RATER 2.0 – FINAL
Python 3.11 recommended
"""

import cv2, mediapipe as mp, numpy as np, math, json, io
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ===================== UTIL =====================
def clamp(v): return max(1, min(10, float(v)))

def mean(lst): return sum(lst) / len(lst) if lst else 0

# ===================== MEDIAPIPE =====================
mp_face = mp.solutions.face_mesh
mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# ===================== ANALYSIS CORE =====================
def analyze_face(img, is_male):
    h, w = img.shape[:2]
    res = mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None

    L = res.multi_face_landmarks[0].landmark
    def d(a,b): return math.dist((L[a].x,L[a].y),(L[b].x,L[b].y))

    # ---------- STRUCTURE ----------
    face_h = d(10,152)
    face_w = d(234,454)
    midface = d(9,2)
    lowerface = d(2,152)
    eye_dist = d(33,263)
    cheekbones = d(127,356)
    jaw_width = d(234,454)
    nose_w = d(98,327)
    mouth_w = d(61,291)

    thirds_balance = clamp(10 - abs(midface/lowerface - 1.05)*10)
    symmetry = clamp(10 - abs(L[33].y - L[263].y)*40)
    jaw_score = clamp((jaw_width/face_h)*12)
    cheekbone_score = clamp((cheekbones/face_w)*14)
    eye_spacing = clamp(10 - abs((eye_dist/face_w)-0.46)*25)
    nose_balance = clamp(10 - abs((nose_w/mouth_w)-0.5)*20)

    harmony = clamp(mean([
        thirds_balance, symmetry, cheekbone_score,
        jaw_score, eye_spacing, nose_balance
    ]))

    # ---------- SKIN ----------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray, cv2.CV_64F).std()
    skin_score = clamp(10 - texture/7)

    # ---------- EYES ----------
    eyelid = abs(L[159].y - L[145].y)
    eye_shape = clamp(5 + eyelid*35)

    # ---------- SEXUAL DIMORPHISM ----------
    if is_male:
        dimorphism = clamp(mean([
            jaw_score*1.2,
            cheekbone_score,
            harmony
        ]))
        dim_label = "Masculinity"
    else:
        dimorphism = clamp(mean([
            eye_shape*1.4,
            symmetry,
            harmony
        ]))
        dim_label = "Femininity"

    # ---------- FACE RATING (OBJECTIVE) ----------
    face_rating = clamp(mean([
        harmony*1.3,
        symmetry,
        jaw_score,
        cheekbone_score,
        thirds_balance
    ]))

    # ---------- ATTRACTIVENESS (PERCEIVED) ----------
    attractiveness = clamp(mean([
        face_rating*1.4,
        dimorphism,
        eye_shape,
        skin_score
    ]))

    # ---------- POTENTIAL ----------
    grooming_gain = (10 - skin_score)*0.6 + (10 - eye_shape)*0.4
    structural_gap = max(0, 10-face_rating)
    potential = clamp(structural_gap*0.7 + grooming_gain*0.3)

    return {
        "Face Rating": round(face_rating,2),
        "Attractiveness": round(attractiveness,2),
        dim_label: round(dimorphism,2),
        "Facial Harmony": round(harmony,2),
        "Symmetry": round(symmetry,2),
        "Jaw": round(jaw_score,2),
        "Cheekbones": round(cheekbone_score,2),
        "Eye Shape": round(eye_shape,2),
        "Skin Quality": round(skin_score,2),
        "Potential": round(potential,2),
        "Image": img
    }

# ===================== PDF =====================
def save_pdf(rep, path):
    c = canvas.Canvas(path, pagesize=A4)
    y = 800
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40,y,"Photo Rater 2.0 Report")
    y -= 20
    c.setFont("Helvetica",10)
    c.drawString(40,y,f"Generated: {datetime.now()}")
    y -= 30
    for k,v in rep.items():
        if k=="Image": continue
        c.drawString(40,y,f"{k}: {v}/10")
        y -= 14
    c.save()

# ===================== GUI =====================
class App:
    def __init__(self,root):
        self.root=root
        root.title("Photo Rater 2.0 (Final)")
        self.img=None
        self.rep=None

        top=tk.Frame(root); top.pack()
        self.gender=tk.StringVar(value="M")
        tk.Radiobutton(top,text="Male",variable=self.gender,value="M").pack(side="left")
        tk.Radiobutton(top,text="Female",variable=self.gender,value="F").pack(side="left")

        tk.Button(root,text="Upload Image",command=self.load).pack()
        tk.Button(root,text="Analyze",command=self.run).pack()
        tk.Button(root,text="Save PDF",command=self.save).pack()

        self.preview=tk.Label(root,bg="#333",width=600,height=400)
        self.preview.pack(pady=6)

        self.out=tk.Text(root,width=80,height=20)
        self.out.pack()

    def load(self):
        p=filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.jpeg")])
        if not p: return
        self.img=cv2.imread(p)
        im=Image.fromarray(cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB))
        im.thumbnail((600,400))
        self.tk=ImageTk.PhotoImage(im)
        self.preview.configure(image=self.tk)

    def run(self):
        if self.img is None: return
        self.rep=analyze_face(self.img,self.gender.get()=="M")
        if not self.rep:
            messagebox.showerror("Error","No face detected")
            return
        self.out.delete(1.0,tk.END)
        for k,v in self.rep.items():
            if k!="Image":
                self.out.insert(tk.END,f"{k}: {v}/10\n")

    def save(self):
        if not self.rep: return
        p=filedialog.asksaveasfilename(defaultextension=".pdf")
        if p:
            save_pdf(self.rep,p)
            messagebox.showinfo("Saved","PDF saved")

# ===================== RUN =====================
if __name__=="__main__":
    root=tk.Tk()
    App(root)
    root.mainloop()
