import cv2
import mediapipe as mp
import numpy as np
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# ===================== MEDIAPIPE =====================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1)

# ===================== HELPERS =====================
def dist(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def clamp(x, a=1, b=10):
    return max(a, min(b, x))

# ===================== ANALYSIS =====================
def analyze_face(img, gender):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        raise ValueError("No face detected.")

    lm = res.multi_face_landmarks[0].landmark

    jaw_w = dist(lm[234], lm[454])
    face_h = dist(lm[10], lm[152])
    cheek_w = dist(lm[50], lm[280])
    eye_l = dist(lm[33], lm[133])
    eye_r = dist(lm[362], lm[263])

    fwhr = jaw_w / face_h
    jaw_score = clamp(fwhr * 12)
    cheek_score = clamp((cheek_w / jaw_w) * 10)
    symmetry = clamp(10 - abs(eye_l - eye_r) * 200)
    eye_score = clamp(((eye_l + eye_r) / 2) * 20)
    face_fat = clamp(10 - (jaw_w / cheek_w) * 4)
    skin = 7.5
    hair = 7.0

    harmony = clamp((jaw_score + cheek_score + symmetry) / 3)
    face_rating = clamp((jaw_score + cheek_score + symmetry + face_fat) / 4)
    attractiveness = clamp((face_rating + eye_score + skin) / 3)
    potential = clamp(attractiveness + 1.5)

    masculinity = femininity = beard = None
    if gender == "Male":
        masculinity = clamp((jaw_score + fwhr * 10) / 2)
        beard = clamp(jaw_score + 1)
    else:
        femininity = clamp((eye_score + cheek_score) / 2)

    flaws = []
    if symmetry < 6: flaws.append("Noticeable asymmetry")
    if jaw_score < 6: flaws.append("Jawline definition weak")
    if face_fat < 6: flaws.append("Face fat imbalance")
    if eye_score < 6: flaws.append("Eye area underdeveloped")

    strengths = []
    if cheek_score > 7: strengths.append("Strong cheekbones")
    if jaw_score > 7: strengths.append("Good jaw structure")
    if symmetry > 7: strengths.append("High symmetry")
    if eye_score > 7: strengths.append("Attractive eye area")

    report = f"""
================ FACE RATER v1.21 =================

Gender: {gender}

----- STRUCTURAL RATINGS -----
Jawline: {jaw_score:.1f}/10
Cheekbones: {cheek_score:.1f}/10
Facial Symmetry: {symmetry:.1f}/10
Facial Harmony: {harmony:.1f}/10
Face Fat Balance: {face_fat:.1f}/10

----- EYES & SURFACE -----
Eye Area: {eye_score:.1f}/10
Skin Quality: {skin:.1f}/10
Hair: {hair:.1f}/10
"""

    if gender == "Male":
        report += f"""
----- MALE TRAITS -----
Masculinity: {masculinity:.1f}/10
Beard Potential: {beard:.1f}/10
"""

    if gender == "Female":
        report += f"""
----- FEMALE TRAITS -----
Femininity: {femininity:.1f}/10
"""

    report += f"""
----- OVERALL -----
Face Rating (Structure): {face_rating:.1f}/10
Attractiveness (Visual): {attractiveness:.1f}/10
Maximum Potential: {potential:.1f}/10

----- STRENGTHS -----
{chr(10).join("• "+s for s in strengths) if strengths else "• None notable"}

----- FLAWS -----
{chr(10).join("• "+f for f in flaws) if flaws else "• Minor / none"}

----- NOTES -----
• Structure ≠ Attractiveness
• Potential assumes grooming, fat %, skincare
• Ratings are analytical, not personal

===================================================
"""
    return report, img

# ===================== GUI =====================
class App:
    def __init__(self, root):
        root.title("Face Rater v1.21")
        root.geometry("950x720")

        self.gender = tk.StringVar(value="Male")

        canvas = tk.Canvas(root)
        scroll = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        self.frame = ttk.Frame(canvas)

        self.frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.frame, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        ttk.Label(self.frame, text="Face Rater v1.21", font=("Arial", 20)).pack(pady=10)

        g = ttk.Frame(self.frame)
        g.pack()
        ttk.Radiobutton(g, text="Male", variable=self.gender, value="Male").pack(side="left", padx=10)
        ttk.Radiobutton(g, text="Female", variable=self.gender, value="Female").pack(side="left", padx=10)

        ttk.Button(self.frame, text="Upload Photo", command=self.upload).pack(pady=5)
        ttk.Button(self.frame, text="Take Webcam Photo", command=self.webcam).pack(pady=5)

        self.img_label = ttk.Label(self.frame)
        self.img_label.pack(pady=10)

        self.text = tk.Text(self.frame, wrap="word", width=105, height=32)
        self.text.pack(pady=10)

    def process(self, img):
        report, img = analyze_face(img, self.gender.get())
        self.text.delete("1.0", tk.END)
        self.text.insert(tk.END, report)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((420, 420))
        tk_img = ImageTk.PhotoImage(img)
        self.img_label.configure(image=tk_img)
        self.img_label.image = tk_img

    def upload(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Invalid image.")
            return
        self.process(img)

    def webcam(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Camera failed.")
            return
        self.process(frame)

# ===================== RUN =====================
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
