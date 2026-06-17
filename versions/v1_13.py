"""
face_analyzer_gui.py

Requirements:
    pip install opencv-python mediapipe numpy pillow reportlab

Run with Python 3.11 (recommended).
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import math
import io
import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# -------------------- Setup MediaPipe --------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# -------------------- Utility functions --------------------
def dist(a, b):
    return math.hypot((a.x - b.x), (a.y - b.y))

def clamp(x, a=1, b=10):
    return max(a, min(b, x))

def meaning(score):
    if score >= 8: return "Excellent"
    if score >= 6.5: return "Good"
    if score >= 5: return "Average"
    return "Below Average"

def pil_from_cv2(cv_img, maxsize=(720,720)):
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    pil.thumbnail(maxsize, Image.LANCZOS)
    return pil

# -------------------- Analysis core --------------------
def analyze_image(cv_image, is_male=None):
    """
    Input: cv_image (BGR numpy array), is_male (True/False/None)
    Output: dict report
    """
    h, w = cv_image.shape[:2]
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    face = results.multi_face_landmarks[0].landmark

    # choose landmarks repeatedly used
    L = face
    # key landmarks (MediaPipe indexes)
    left_jaw = L[234]; right_jaw = L[454]
    chin = L[152]; forehead = L[10]
    left_cheek = L[93]; right_cheek = L[323]
    left_eye_inner = L[133]; left_eye_outer = L[33]
    right_eye_inner = L[362]; right_eye_outer = L[263]
    nose_tip = L[1]
    mouth_left = L[61]; mouth_right = L[291]

    # Measurements (normalized because mediapipe uses relative coords)
    jaw_w = math.hypot(left_jaw.x - right_jaw.x, left_jaw.y - right_jaw.y)
    face_h = math.hypot(forehead.x - chin.x, forehead.y - chin.y)
    face_w = math.hypot(left_cheek.x - right_cheek.x, left_cheek.y - right_cheek.y)
    eye_dist = math.hypot(left_eye_outer.x - right_eye_outer.x, left_eye_outer.y - right_eye_outer.y)
    mouth_w = math.hypot(mouth_left.x - mouth_right.x, mouth_left.y - mouth_right.y)

    # Ratios
    fwhr = face_w / face_h if face_h > 0 else 0
    jaw_ratio = jaw_w / face_h if face_h > 0 else 0
    eye_ratio = eye_dist / face_w if face_w > 0 else 0
    mouth_face_ratio = mouth_w / face_w if face_w > 0 else 0

    # Core scores mapped to 1-10
    jaw_score = clamp(jaw_ratio * 12)
    symmetry_score = clamp(10 - abs((left_eye_outer.x - right_eye_outer.x)) * 30)  # rough
    eye_score = clamp(10 - abs(eye_ratio - 0.46) * 20)
    fwhr_score = clamp((1 / (abs(fwhr - 0.45) + 0.1)) * 1.2)  # center ideal near 0.45-0.5 in normalized space
    canthal_tilt_score = clamp(((right_eye_outer.y - left_eye_outer.y) * 100) + 5)

    # Skin scoring: brightness + smoothness heuristic
    # Convert to HSV and inspect V and small-scale std for blemishes
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(float) / 255.0
    brightness = v.mean()
    small_std = cv2.Laplacian(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).std()
    # lower laplacian std often corresponds to smoother skin (over-simplified)
    skin_score = clamp((brightness * 4) + (10 / (small_std + 1)))

    # Hairline / hair rating (heuristic): measure forehead length relative to face height
    forehead_to_chin = math.hypot(forehead.x - chin.x, forehead.y - chin.y)
    forehead_ratio = (forehead.y - nose_tip.y) if forehead_to_chin>0 else 0  # not robust, but heuristic
    # we'll use face_h and forehead distance as rough hairline indicator
    hair_score = clamp(10 - (abs(face_h * 0.18 - (forehead.y - nose_tip.y)) * 40))  # heuristic

    # Beard potential: for males, jaw strength + lower face prominence
    beard_potential = clamp((jaw_score * 1.3 + fwhr_score * 0.7) / 2.0)

    # Face fat heuristic: if lower face width vs midface is high -> leaner; we use mouth/cheek distances
    lower_face_width = math.hypot(L[152].x - L[152].x, L[152].y - L[152].y)  # dummy (0)
    # Instead use ratio of cheek prominence to jaw width
    cheek_prom = math.hypot(L[10].x - L[93].x, L[10].y - L[93].y)
    face_fat = clamp(10 - (jaw_score + (face_w/face_h)*5)/3.0)  # heuristic: higher jaw => less fat

    # Face shape detection based on proportions
    # Use ratios face_w/face_h and jaw prominence
    fs_ratio = face_w / face_h if face_h>0 else 0
    if fs_ratio > 0.96:
        face_shape = "Round"
    elif fs_ratio < 0.78:
        face_shape = "Long/Oblong"
    else:
        if jaw_score > 7.5:
            face_shape = "Square"
        else:
            face_shape = "Oval"

    # Attractiveness overall synthetic score (normalized comp of several metrics)
    overall_score = clamp((jaw_score*1.1 + symmetry_score*1.2 + eye_score*1.1 + skin_score*1.0 + fwhr_score*0.8) / 5.2)
    # Attractiveness percentile mapping (heuristic)
    # We map 1-10 score to percentiles roughly: >9 -> 99th, >8->90th, >7->75th, >6->60th, >5->45th, else linear
    if overall_score >= 9:
        percentile = 99
    elif overall_score >= 8:
        percentile = 90
    elif overall_score >= 7:
        percentile = 75
    elif overall_score >= 6:
        percentile = 60
    elif overall_score >= 5:
        percentile = 45
    else:
        percentile = int(max(1, (overall_score / 10.0) * 40))

    # Gender-specific scores: require is_male flag passed (True/False)
    masculinity = None
    femininity = None
    if is_male is not None:
        if is_male:
            masculinity = clamp((jaw_score*1.4 + fwhr_score*0.9 + canthal_tilt_score*0.7) / 3.0)
        else:
            femininity = clamp((eye_score*1.4 + symmetry_score*1.2 + (10 - jaw_score)*0.8) / 3.4)

    # POSITIVES / NEGATIVES logic
    positives = []
    negatives = []
    if jaw_score > 7: positives.append("Strong jawline")
    if symmetry_score > 7: positives.append("Good facial symmetry")
    if eye_score > 7: positives.append("Attractive eye proportions")
    if skin_score > 7: positives.append("Clear / even skin tone")
    if hair_score > 6: positives.append("Good hairline / hair density (heuristic)")

    if jaw_score < 5: negatives.append("Weak jaw definition")
    if symmetry_score < 5: negatives.append("Noticeable asymmetry")
    if skin_score < 5: negatives.append("Uneven skin / possible texture issues")
    if face_fat > 7: negatives.append("Facial roundness/softness present")

    # Compose report
    report = {
        "jaw_score": round(jaw_score,1),
        "symmetry_score": round(symmetry_score,1),
        "eye_score": round(eye_score,1),
        "fwhr_score": round(fwhr_score,1),
        "canthal_tilt_score": round(canthal_tilt_score,1),
        "skin_score": round(skin_score,1),
        "hair_score": round(hair_score,1),
        "beard_potential": round(beard_potential,1),
        "face_fat": round(face_fat,1),
        "face_shape": face_shape,
        "fwhr": round(fwhr,3),
        "overall_score": round(overall_score,2),
        "percentile": percentile,
        "positives": positives,
        "negatives": negatives,
        "masculinity": round(masculinity,1) if masculinity is not None else None,
        "femininity": round(femininity,1) if femininity is not None else None,
        "image": cv_image.copy()
    }

    return report

# -------------------- PDF saving --------------------
def save_report_pdf(report, out_path):
    c = canvas.Canvas(out_path, pagesize=A4)
    W, H = A4
    margin = 40
    x = margin
    y = H - margin

    title = "Detailed Face Analysis Report"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20

    # embed image (resize to fit)
    img = report.get("image")
    if img is not None:
        pil = pil_from_cv2(img, maxsize=(400,400))
        bio = io.BytesIO()
        pil.save(bio, format="PNG")
        bio.seek(0)
        # draw at right side
        iw, ih = pil.size
        img_x = W - margin - iw
        c.drawImage(bio, img_x, y-ih+20, width=iw, height=ih)
    y -= 10 + (ih if img is not None else 0)

    # Text body
    def line(s):
        nonlocal y
        if y < 100:
            c.showPage()
            y = H - margin
        c.drawString(x, y, s)
        y -= 14

    line(f"Overall Score: {report['overall_score']}/10")
    line(f"Estimated Attractiveness Percentile: {report['percentile']}th")
    line(" ")
    line("STRUCTURE SCORES:")
    line(f" - Jawline: {report['jaw_score']}/10")
    line(f" - Symmetry: {report['symmetry_score']}/10")
    line(f" - Eye area: {report['eye_score']}/10")
    line(f" - FWHR score: {report['fwhr_score']}/10")
    line(f" - Canthal tilt score: {report['canthal_tilt_score']}/10")
    line(" ")
    line("SKIN / HAIR / BEARD:")
    line(f" - Skin score: {report['skin_score']}/10")
    line(f" - Hairline / hair score: {report['hair_score']}/10")
    line(f" - Beard potential: {report['beard_potential']}/10")
    line(" ")
    line(f"Face shape: {report['face_shape']}, FWHR: {report['fwhr']}")
    if report.get("masculinity") is not None:
        line(f"Masculinity score: {report['masculinity']}/10")
    if report.get("femininity") is not None:
        line(f"Femininity score: {report['femininity']}/10")
    line(" ")
    line("POSITIVES:")
    for p in report['positives']:
        line(" - " + p)
    line("NEGATIVES:")
    for n in report['negatives']:
        line(" - " + n)

    c.save()

# -------------------- GUI --------------------
class FaceApp:
    def __init__(self, master):
        self.master = master
        master.title("Looksmax Analyzer - Desktop")
        self.cv_image = None
        self.report = None
        self.is_male = None

        # Top frame: gender selection
        top = tk.Frame(master)
        top.pack(padx=8, pady=6, anchor="w")
        tk.Label(top, text="Select biological sex:").pack(side="left")
        self.gender_var = tk.StringVar(value="1")
        tk.Radiobutton(top, text="Male", variable=self.gender_var, value="1").pack(side="left")
        tk.Radiobutton(top, text="Female", variable=self.gender_var, value="2").pack(side="left")

        # Buttons
        btn_frame = tk.Frame(master)
        btn_frame.pack(padx=8, pady=6, anchor="w")
        tk.Button(btn_frame, text="Upload Photo", command=self.upload_image).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Webcam Capture", command=self.capture_image).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Analyze", command=self.analyze).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Save PDF", command=self.save_pdf).pack(side="left", padx=4)

        # Preview canvas
        self.preview_label = tk.Label(master, text="No image loaded", width=80, height=24, bg="#222", fg="#ddd")
        self.preview_label.pack(padx=8, pady=6)

        # Report text
        self.report_text = tk.Text(master, width=90, height=18)
        self.report_text.pack(padx=8, pady=6)

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot open image")
            return
        self.cv_image = img
        pil = pil_from_cv2(img)
        self.display_pil(pil)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return
        messagebox.showinfo("Capture", "Press SPACE in the video window to capture, ESC to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Webcam - Press SPACE to capture", frame)
            k = cv2.waitKey(1)
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            if k == 32:
                cap.release()
                cv2.destroyAllWindows()
                self.cv_image = frame.copy()
                pil = pil_from_cv2(self.cv_image)
                self.display_pil(pil)
                return

    def display_pil(self, pil):
        self.pil_image = pil
        self.tkimage = ImageTk.PhotoImage(pil)
        self.preview_label.config(image=self.tkimage, text="")

    def analyze(self):
        if self.cv_image is None:
            messagebox.showwarning("No image", "Please upload or capture an image first.")
            return
        # gender choice
        self.is_male = True if self.gender_var.get() == "1" else False
        # run analysis
        rpt = analyze_image(self.cv_image, is_male=self.is_male)
        if rpt.get("error"):
            messagebox.showerror("Analysis Error", rpt["error"])
            return
        self.report = rpt
        self.show_report_text(rpt)

    def show_report_text(self, rpt):
        self.report_text.delete(1.0, tk.END)
        lines = []
        lines.append("DETAILED FACE ANALYSIS REPORT")
        lines.append(f"Overall score: {rpt['overall_score']}/10  ({rpt['percentile']}th percentile)")
        lines.append("")
        lines.append("STRUCTURE & RATIOS")
        lines.append(f" - Face shape: {rpt['face_shape']}")
        lines.append(f" - FWHR: {rpt['fwhr']}")
        lines.append(f" - Jawline: {rpt['jaw_score']}/10")
        lines.append(f" - Symmetry: {rpt['symmetry_score']}/10")
        lines.append(f" - Eye area: {rpt['eye_score']}/10")
        lines.append(f" - Canthal tilt: {rpt['canthal_tilt_score']}/10")
        lines.append("")
        lines.append("SKIN / HAIR / BEARD")
        lines.append(f" - Skin score: {rpt['skin_score']}/10")
        lines.append(f" - Hairline/hair score: {rpt['hair_score']}/10")
        lines.append(f" - Beard potential: {rpt['beard_potential']}/10")
        lines.append(f" - Face fat estimate: {rpt['face_fat']}/10")
        lines.append("")
        if rpt.get("masculinity") is not None:
            lines.append(f"Masculinity score: {rpt['masculinity']}/10")
        if rpt.get("femininity") is not None:
            lines.append(f"Femininity score: {rpt['femininity']}/10")
        lines.append("")
        lines.append("POSITIVES:")
        for p in rpt['positives']:
            lines.append("  - " + p)
        lines.append("NEGATIVES:")
        for n in rpt['negatives']:
            lines.append("  - " + n)
        lines.append("")
        lines.append("Advice & Notes:")
        lines.append(" - Take photos in neutral lighting, head straight, camera at eye level for best accuracy.")
        lines.append(" - Scores are heuristics (not medical). Grooming, hairstyle, and confidence matter a lot.")
        self.report_text.insert(tk.END, "\n".join(lines))

    def save_pdf(self):
        if self.report is None:
            messagebox.showwarning("No report", "Please analyze an image first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".pdf",
                                           filetypes=[("PDF","*.pdf")],
                                           title="Save report as PDF")
        if not out:
            return
        try:
            save_report_pdf(self.report, out)
            messagebox.showinfo("Saved", f"Report saved to {out}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))


# -------------------- Run GUI --------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceApp(root)
    root.mainloop()
