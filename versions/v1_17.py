"""
photo_rater_1_17.py
Requirements:
  py -3.11 -m pip install opencv-python mediapipe numpy pillow reportlab
Run with Python 3.11 recommended.
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import io
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# -------------------- Utilities --------------------
def clamp(x, lo=1, hi=10):
    return max(lo, min(hi, x))

def meaning(v):
    if v >= 8: return "Excellent"
    if v >= 6.5: return "Good"
    if v >= 5: return "Average"
    return "Below Average"

def pil_from_cv2(cv_img, maxsize=(720,720)):
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil.thumbnail(maxsize, Image.LANCZOS)
    return pil

# -------------------- MediaPipe --------------------
mp_face = mp.solutions.face_mesh
face_mesh_proc = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# -------------------- Core analysis --------------------
def analyze_image(cv_image, is_male=None):
    """Return a dict report for one image (BGR numpy)."""
    h, w = cv_image.shape[:2]
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    res = face_mesh_proc.process(rgb)
    if not res.multi_face_landmarks:
        return {"error":"No face detected"}

    L = res.multi_face_landmarks[0].landmark

    # helper distance in normalized coordinates
    def d(i,j):
        return math.hypot(L[i].x - L[j].x, L[i].y - L[j].y)

    # key landmarks indexes (MediaPipe)
    # jaw: 234 (left), 454 (right), chin 152, forehead 10
    # cheeks/zygos: 127, 356 (approx)
    # eyes: outer 33/263 inner 133/362
    jaw_w = d(234, 454)
    face_h = d(10, 152) if d(10,152)>0 else 1e-6
    cheek_w = d(93, 323)
    zyg_dist = d(127, 356)  # cheekbone spread
    eye_dist = d(33, 263)
    left_eye_height = abs(L[159].y - L[145].y)  # eyelid openness proxy

    # Ratios
    fwhr = cheek_w / face_h
    jaw_ratio = jaw_w / face_h
    eye_ratio = eye_dist / cheek_w if cheek_w>0 else 0
    zyg_ratio = zyg_dist / cheek_w if cheek_w>0 else 0

    # Base scores (1-10)
    jaw_score = clamp(jaw_ratio * 12)
    symmetry_score = clamp(10 - abs(L[33].y - L[263].y)*30)  # rough symmetry via eyes vertical
    eye_score = clamp(10 - abs(eye_ratio - 0.46)*20)
    fwhr_score = clamp((1/(abs(fwhr - 0.45)+0.08))*1.2)
    zyg_score = clamp(zyg_ratio * 12)  # cheekbone prominence estimate
    canthal_tilt_score = clamp(((L[263].y - L[33].y)*100) + 5)

    # Skin: brightness + texture + blemish heuristic
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    v = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)[:,:,2].astype(float)/255.0
    brightness = float(v.mean())
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture_std = float(lap.std())
    texture_score = clamp(10 - (texture_std / 8))  # smoother -> higher
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    detail = cv2.absdiff(gray, blur)
    _, bm = cv2.threshold(detail, int(max(10, np.percentile(detail,90))), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blemles = sum(1 for c in contours if 10 < cv2.contourArea(c) < 2000)
    blemish_score = clamp(10 - (blemles/6))
    skin_score = clamp(brightness*4 + texture_score*0.6 + blemish_score*0.8)

    # Hair score heuristic (forehead ratio)
    forehead_to_nose = abs(L[10].y - L[1].y)
    hair_score = clamp(10 - abs(forehead_to_nose*15 - 2))

    # Beard potential (male)
    beard_potential = clamp((jaw_score*1.25 + fwhr_score*0.6)/2.0)

    # Face fat (heuristic)
    face_fat = clamp(10 - (jaw_score + (cheek_w/face_h)*5)/3.0)

    # Facial Harmony (structural balance)
    # Harmony uses fwhr vs an ideal and cheek/jaw balance
    ideal_fwhr = 1.9
    harmony = clamp(10 - abs(fwhr - ideal_fwhr)*6 + (zyg_score-5)*0.15)

    # Face Rating (objective structural)
    face_rating = clamp(
        jaw_score*0.22 +
        symmetry_score*0.24 +
        harmony*0.28 +
        zyg_score*0.13 +
        eye_score*0.13
    )

    # Sexual dimorphism (masculinity/femininity)
    if is_male is None:
        dimorphism = None
    else:
        if is_male:
            dimorphism = clamp((jaw_score*1.3 + fwhr_score*0.9 + canthal_tilt_score*0.7)/3.0)
            dim_label = "Masculinity"
        else:
            dimorphism = clamp((eye_score*1.4 + symmetry_score*1.2 + (10 - jaw_score)*0.8)/3.4)
            dim_label = "Femininity"

    # Attractiveness (perceived)
    attractiveness = clamp(
        face_rating*0.55 +
        (dimorphism if dimorphism is not None else face_rating)*0.3 +
        eye_score*0.1 +
        canthal_tilt_score*0.05
    )

    # Potential face (how much room to improve with grooming/skin)
    # Higher when face_rating low but beard/hair/skin can change easily:
    structural_gap = max(0, 10 - face_rating)
    modifiable = (10 - skin_score) * 0.6 + (10 - beard_potential) * 0.3 + (10 - hair_score) * 0.1
    potential_score = clamp(min(10, structural_gap*0.7 + modifiable*0.3))

    # Compose report
    report = {
        "jaw_score": round(jaw_score,1),
        "symmetry_score": round(symmetry_score,1),
        "eye_score": round(eye_score,1),
        "fwhr_score": round(fwhr_score,1),
        "zyg_score": round(zyg_score,1),
        "canthal_tilt_score": round(canthal_tilt_score,1),
        "skin_score": round(skin_score,1),
        "texture_std": round(texture_std,2),
        "blemish_count": int(blemles),
        "hair_score": round(hair_score,1),
        "beard_potential": round(beard_potential,1),
        "face_fat": round(face_fat,1),
        "harmony": round(harmony,1),
        "face_rating": round(face_rating,1),
        "dimorphism": round(dimorphism,1) if dimorphism is not None else None,
        "dimorphism_label": dim_label if is_male is not None else None,
        "attractiveness": round(attractiveness,1),
        "potential_score": round(potential_score,1),
        "positives": [],
        "negatives": [],
        "landmarks": [{"x":float(x.x),"y":float(x.y),"z":float(x.z)} for x in L],
        "image": cv_image.copy()
    }

    # Positives / negatives
    if report["jaw_score"] > 7: report["positives"].append("Strong jawline")
    if report["symmetry_score"] > 7: report["positives"].append("Good symmetry")
    if report["zyg_score"] > 7: report["positives"].append("Prominent cheekbones")
    if report["skin_score"] > 7: report["positives"].append("Clear skin")
    if report["hair_score"] > 6: report["positives"].append("Healthy hairline (approx)")

    if report["jaw_score"] < 5: report["negatives"].append("Weak jaw definition")
    if report["symmetry_score"] < 5: report["negatives"].append("Noticeable asymmetry")
    if report["skin_score"] < 5: report["negatives"].append("Skin texture/blemishes")
    if report["potential_score"] > 6: report["negatives"].append("High improvement potential via grooming/skin")

    return report

# -------------------- Save PDF & JSON --------------------
def save_report_pdf(report, out_path):
    c = canvas.Canvas(out_path, pagesize=A4)
    W, H = A4
    margin = 36
    x = margin
    y = H - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Photo Rater 1.17 - Detailed Report")
    y -= 22
    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 18

    # image
    img = report.get("image")
    ih = 0
    if img is not None:
        pil = pil_from_cv2(img, maxsize=(360,360))
        bio = io.BytesIO()
        pil.save(bio, format="PNG")
        bio.seek(0)
        iw, ih = pil.size
        c.drawImage(bio, W - margin - iw, y - ih + 12, width=iw, height=ih)
    y -= ih + 6

    def line(s):
        nonlocal y
        if y < 80:
            c.showPage(); y = H - margin
        c.drawString(x, y, s); y -= 12

    line(f"Face Rating (structure): {report['face_rating']}/10")
    line(f"Attractiveness: {report['attractiveness']}/10")
    if report.get("dimorphism") is not None:
        line(f"{report['dimorphism_label']}: {report['dimorphism']}/10")
    line(f"Potential Score: {report['potential_score']}/10")
    line("")
    line("STRUCTURAL SCORES:")
    line(f" - Jaw: {report['jaw_score']}/10")
    line(f" - Symmetry: {report['symmetry_score']}/10")
    line(f" - Cheekbones(Zyg): {report['zyg_score']}/10")
    line(f" - Eye area: {report['eye_score']}/10")
    line(f" - Harmony: {report['harmony']}/10")
    line("")
    line("SKIN / HAIR / BEARD:")
    line(f" - Skin score: {report['skin_score']}/10 (texture std: {report['texture_std']})")
    line(f" - Blemish count: {report['blemish_count']}")
    line(f" - Hair score: {report['hair_score']}/10")
    line(f" - Beard potential: {report['beard_potential']}/10")
    line("")
    line("POSITIVES:")
    for p in report['positives']: line(" - " + p)
    line("NEGATIVES:")
    for n in report['negatives']: line(" - " + n)
    c.save()

def save_report_json(report, out_path):
    # remove image binary
    doc = {k:v for k,v in report.items() if k!="image"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

# -------------------- GUI Application --------------------
class App:
    def __init__(self, master):
        self.master = master
        master.title("Photo Rater 1.17")
        self.cv_image = None
        self.report = None
        self.overlay_on = tk.BooleanVar(value=False)
        self.multi_n = tk.IntVar(value=3)

        # top controls
        top = tk.Frame(master); top.pack(padx=6, pady=6, anchor="w")
        tk.Label(top, text="Biological sex:").pack(side="left")
        self.gender_var = tk.StringVar(value="1")
        tk.Radiobutton(top, text="Male", variable=self.gender_var, value="1").pack(side="left")
        tk.Radiobutton(top, text="Female", variable=self.gender_var, value="2").pack(side="left")

        btns = tk.Frame(master); btns.pack(padx=6, pady=6, anchor="w")
        tk.Button(btns, text="Upload Photo", command=self.upload_image).pack(side="left", padx=4)
        tk.Button(btns, text="Webcam Capture", command=self.capture_image).pack(side="left", padx=4)
        tk.Button(btns, text="Multi-shot Avg", command=self.multi_shot).pack(side="left", padx=4)
        tk.Button(btns, text="Analyze", command=self.analyze).pack(side="left", padx=4)
        tk.Button(btns, text="Save PDF", command=self.save_pdf).pack(side="left", padx=4)
        tk.Button(btns, text="Save JSON", command=self.save_json).pack(side="left", padx=4)

        opts = tk.Frame(master); opts.pack(padx=6, pady=3, anchor="w")
        tk.Checkbutton(opts, text="Show landmark overlay", variable=self.overlay_on, command=self.redraw).pack(side="left")
        tk.Label(opts, text="Multi-shot N:").pack(side="left", padx=(8,0))
        tk.Spinbox(opts, from_=1, to=7, width=3, textvariable=self.multi_n).pack(side="left")

        self.preview = tk.Label(master, text="No image", width=80, height=24, bg="#222", fg="#ddd")
        self.preview.pack(padx=6, pady=6)
        self.text = tk.Text(master, width=100, height=20)
        self.text.pack(padx=6, pady=6)

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
        if not path: return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot open image")
            return
        self.cv_image = img
        self.redraw()

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return
        messagebox.showinfo("Capture", "Press SPACE to capture, ESC to cancel")
        while True:
            ret, frame = cap.read()
            if not ret: continue
            cv2.imshow("Webcam - Press SPACE", frame)
            k = cv2.waitKey(1)
            if k == 27:
                cap.release(); cv2.destroyAllWindows(); return
            if k == 32:
                cap.release(); cv2.destroyAllWindows()
                self.cv_image = frame.copy(); self.redraw(); return

    def multi_shot(self):
        n = int(self.multi_n.get())
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam"); return
        messagebox.showinfo("Multi-shot", f"Capture {n} shots, press SPACE for each.")
        imgs = []
        for i in range(n):
            while True:
                ret, frame = cap.read()
                if not ret: continue
                cv2.imshow(f"Shot {i+1}/{n} - SPACE", frame)
                k = cv2.waitKey(1)
                if k == 27:
                    cap.release(); cv2.destroyAllWindows(); return
                if k == 32:
                    imgs.append(frame.copy()); break
        cap.release(); cv2.destroyAllWindows()
        avg = np.mean(np.stack([cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) for im in imgs]), axis=0)
        avg_bgr = cv2.cvtColor(avg.astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.cv_image = avg_bgr; self.redraw()
        messagebox.showinfo("Multi-shot", "Averaged image created.")

    def redraw(self):
        if self.cv_image is None:
            self.preview.config(text="No image", image="", bg="#222"); return
        img = self.cv_image.copy()
        if self.overlay_on.get():
            res = face_mesh_proc.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                mp_drawing.draw_landmarks(image=img, landmark_list=res.multi_face_landmarks[0],
                                          connections=mp_face.FACEMESH_TESSELATION,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,80,80), thickness=1))
        pil = pil_from_cv2(img, maxsize=(640,480))
        self.tkimg = ImageTk.PhotoImage(pil)
        self.preview.config(image=self.tkimg, text="")

    def analyze(self):
        if self.cv_image is None:
            messagebox.showwarning("No image", "Upload or capture first")
            return
        is_male = True if self.gender_var.get()=="1" else False
        rpt = analyze_image(self.cv_image, is_male=is_male)
        if rpt.get("error"):
            messagebox.showerror("Analysis error", rpt["error"]); return
        self.report = rpt
        self.show_report(rpt)

    def show_report(self, rpt):
        self.text.delete(1.0, tk.END)
        lines = []
        lines.append("PHOTO RATER 1.17 - REPORT")
        lines.append(f"Face Rating (structure): {rpt['face_rating']}/10")
        lines.append(f"Attractiveness: {rpt['attractiveness']}/10")
        if rpt.get("dimorphism") is not None:
            lines.append(f"{rpt['dimorphism_label']}: {rpt['dimorphism']}/10")
        lines.append(f"Potential Score: {rpt['potential_score']}/10")
        lines.append("")
        lines.append("STRUCTURAL SCORES:")
        lines.append(f" - Jaw: {rpt['jaw_score']}/10")
        lines.append(f" - Symmetry: {rpt['symmetry_score']}/10")
        lines.append(f" - Cheekbones (zyg): {rpt['zyg_score']}/10")
        lines.append(f" - Eye area: {rpt['eye_score']}/10")
        lines.append(f" - Harmony: {rpt['harmony']}/10")
        lines.append("")
        lines.append("SKIN / HAIR:")
        lines.append(f" - Skin score: {rpt['skin_score']}/10 (texture std: {rpt['texture_std']})")
        lines.append(f" - Blemish count: {rpt['blemish_count']}")
        lines.append(f" - Hair score: {rpt['hair_score']}/10")
        lines.append(f" - Beard potential: {rpt['beard_potential']}/10")
        lines.append("")
        lines.append("POSITIVES:")
        for p in rpt['positives']: lines.append("  - " + p)
        lines.append("NEGATIVES:")
        for n in rpt['negatives']: lines.append("  - " + n)
        lines.append("")
        lines.append("SUGGESTIONS (HIGH-LEVEL):")
        if rpt['skin_score'] < 6:
            lines.append(" - Focus on skin routine: exfoliation, hydration, consult dermatologist for persistent issues.")
        if rpt['jaw_score'] < 6 and rpt['beard_potential'] > 6:
            lines.append(" - Try light stubble or beard styles to enhance jaw definition.")
        if rpt['face_rating'] < 6:
            lines.append(" - Consider posture/weight improvements, grooming and hairline style adjustments to improve perceived structure.")
        lines.append("")
        self.text.insert(tk.END, "\n".join(lines))

    def save_pdf(self):
        if not hasattr(self, "report") or self.report is None:
            messagebox.showwarning("No report", "Analyze an image first"); return
        out = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF","*.pdf")])
        if not out: return
        save_report_pdf(self.report, out)
        messagebox.showinfo("Saved", f"Saved PDF to {out}")

    def save_json(self):
        if not hasattr(self, "report") or self.report is None:
            messagebox.showwarning("No report", "Analyze an image first"); return
        out = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not out: return
        save_report_json(self.report, out)
        messagebox.showinfo("Saved", f"Saved JSON to {out}")

# -------------------- Run --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
