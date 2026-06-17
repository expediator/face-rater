"""
photo_rater_1_20_fixed.py
Run with Python 3.11 recommended.

Requires:
    pip install opencv-python mediapipe numpy pillow reportlab
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import json
import io
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# -------------------- Utilities --------------------
def clamp(v, lo=1.0, hi=10.0):
    try:
        v = float(v)
    except:
        return lo
    return max(lo, min(hi, v))

def mean(vals):
    return sum(vals) / len(vals) if vals else 0

def percentile_from_score(score):
    if score >= 9: return 99
    if score >= 8: return 90
    if score >= 7: return 75
    if score >= 6: return 60
    if score >= 5: return 45
    return int(max(1, (score / 10.0) * 40))

# -------------------- MediaPipe --------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
mp_draw = mp.solutions.drawing_utils

# landmark index aliases used in program (MediaPipe mesh)
LM = {
    "jaw_left": 234, "jaw_right": 454, "chin": 152, "forehead": 10,
    "left_cheek": 127, "right_cheek": 356,
    "left_eye_outer": 33, "right_eye_outer": 263,
    "left_eye_top": 159, "left_eye_bottom": 145,
    "right_eye_top": 386, "right_eye_bottom": 374,
    "nose_tip": 1, "nose_left": 98, "nose_right": 327,
    "mouth_left": 61, "mouth_right": 291,
    "mid_forehead": 9, "mid_nose": 2
}

# -------------------- Core analysis --------------------
def analyze_single_image(cv_image, is_male=None):
    """
    Analyze a BGR OpenCV image. Returns a report dict.
    """
    h, w = cv_image.shape[:2]
    rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"error": "No face detected"}

    landmarks = results.multi_face_landmarks[0].landmark

    def nd(a, b):
        return math.dist((landmarks[a].x, landmarks[a].y), (landmarks[b].x, landmarks[b].y))

    # basic normalized measures
    jaw_w = nd(LM["jaw_left"], LM["jaw_right"])
    face_h = nd(LM["forehead"], LM["chin"]) or 1e-6
    cheek_w = nd(LM["left_cheek"], LM["right_cheek"]) or 1e-6
    eye_d = nd(LM["left_eye_outer"], LM["right_eye_outer"])
    mouth_w = nd(LM["mouth_left"], LM["mouth_right"]) or 1e-6
    nose_w = nd(LM["nose_left"], LM["nose_right"]) or 1e-6

    # ratios and heuristic scores
    fwh_ratio = cheek_w / face_h
    jaw_ratio = jaw_w / face_h
    eye_ratio = eye_d / cheek_w
    mouth_ratio = mouth_w / cheek_w
    nose_mouth_ratio = nose_w / mouth_w

    jaw_score = clamp(jaw_ratio * 12)
    symmetry_score = clamp(10 - abs(landmarks[LM["left_eye_outer"]].y - landmarks[LM["right_eye_outer"]].y) * 40)
    cheekbone_score = clamp((nd(127, 356) / cheek_w) * 12)
    eye_spacing_score = clamp(10 - abs(eye_ratio - 0.46) * 25)
    mouth_balance = clamp(10 - abs(mouth_ratio - 0.35) * 25)
    nose_balance = clamp(10 - abs(nose_mouth_ratio - 0.25) * 25)

    # thirds (upper/mid/lower)
    upper_third = nd(LM["forehead"], LM["mid_forehead"])
    mid_third = nd(LM["mid_forehead"], LM["mid_nose"])
    lower_third = nd(LM["mid_nose"], LM["chin"]) or 1e-6
    thirds_score = clamp(10 - abs((upper_third / mid_third) - 1.0) * 6)

    # canthal tilt and eye openness
    canthal_raw = (landmarks[LM["right_eye_outer"]].y - landmarks[LM["left_eye_outer"]].y)
    canthal_score = clamp(canthal_raw * 100 + 5)

    left_eye_open = abs(landmarks[LM["left_eye_top"]].y - landmarks[LM["left_eye_bottom"]].y)
    right_eye_open = abs(landmarks[LM["right_eye_top"]].y - landmarks[LM["right_eye_bottom"]].y)
    eye_openness = clamp(5 + (left_eye_open + right_eye_open) / 2.0 * 40)

    # skin texture and blemish heuristics
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    texture_std = float(abs(lap).std())
    texture_score = clamp(10 - texture_std / 8.0)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    detail = cv2.absdiff(gray, blur)
    thr = int(max(10, np.percentile(detail, 90)))
    _, blem = cv2.threshold(detail, thr, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(blem.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blem_count = sum(1 for c in contours if 6 < cv2.contourArea(c) < 2000)
    blemish_score = clamp(10 - (blem_count / 6.0))

    skin_score = clamp(texture_score * 0.6 + blemish_score * 0.4)

    # face fat heuristic
    fat_index = (cheek_w / jaw_w) * 5 if jaw_w > 0 else 5
    face_fat = clamp(10 - min(9, fat_index))

    # lip fullness (simple)
    lip_fullness = clamp(mouth_w / (nose_w + 1e-6) * 4)

    # face shape
    fs_ratio = cheek_w / face_h
    if fs_ratio > 0.96:
        face_shape = "Round"
    elif fs_ratio < 0.78:
        face_shape = "Long/Oblong"
    else:
        face_shape = "Square" if jaw_score > 7 else "Oval"

    # cheek prominence
    cheek_prom = cheekbone_score

    # harmony and face rating
    harmony = clamp(mean([thirds_score, symmetry_score, cheekbone_score, jaw_score, nose_balance]))
    face_rating = clamp(mean([harmony * 1.2, symmetry_score, cheekbone_score, jaw_score, thirds_score]))

    # dimorphism
    dimorphism = None
    dim_label = None
    if is_male is not None:
        if is_male:
            dimorphism = clamp(mean([jaw_score * 1.3, cheekbone_score, canthal_score]))
            dim_label = "Masculinity"
        else:
            dimorphism = clamp(mean([eye_openness * 1.3, symmetry_score, (10 - jaw_score)]))
            dim_label = "Femininity"

    # attractiveness (perceived)
    attractiveness = clamp(mean([face_rating * 1.2, (dimorphism if dimorphism is not None else face_rating) * 0.6,
                                 skin_score * 0.9, eye_openness * 0.4]))

    # potential (max attainable) = face_rating + improvable parts (skin, eye_openness, reduce fat)
    improvable = mean([10 - skin_score, 10 - eye_openness, 10 - face_fat])
    potential = clamp(min(10.0, face_rating + improvable * 0.6))

    # percentiles
    face_percentile = percentile_from_score(face_rating)
    attr_percentile = percentile_from_score(attractiveness)
    potential_percentile = percentile_from_score(potential)

    # confidence based on nose x position roughly (if centered, high)
    nose_x = landmarks[LM["nose_tip"]].x
    angle_offset = abs(nose_x - 0.5)
    if angle_offset < 0.04: confidence = "High"
    elif angle_offset < 0.08: confidence = "Medium"
    else: confidence = "Low"

    # asymmetry simple metric
    pairs = [(33, 263), (133, 362), (61, 291), (234, 454)]
    asym_vals = [abs(landmarks[a].x - (1 - landmarks[b].x)) for (a, b) in pairs]
    asymmetry_score = clamp(10 - mean(asym_vals) * 50)

    # suggestions rules
    suggestions = []
    if skin_score < 6:
        suggestions.append("Improve skin routine: cleanse, exfoliate, hydrate; consult dermatologist for persistent issues.")
    if jaw_score < 6:
        suggestions.append("Try grooming (stubble/beard) or hairstyles that increase perceived jaw definition.")
    if cheek_prom < 6:
        suggestions.append("Facial contouring (makeup or cosmetic) or reduce body fat to enhance cheekbones.")
    if eye_openness < 6:
        suggestions.append("Improve sleep, hydration; cosmetics or non-invasive options can help eyelid appearance.")
    if face_fat > 7:
        suggestions.append("Lifestyle/weight management can reduce facial roundness and improve structure.")
    if eye_spacing_score > 7:
        suggestions.append("Wide-set eyes: thicker frames/goggles suit you.")
    elif eye_spacing_score < 4.5:
        suggestions.append("Close-set eyes: narrow bridge frames suit you.")

    # hairstyle & glasses suggestions
    if face_shape == "Round":
        hairstyle = "Add height on top, short sides"
    elif face_shape == "Square":
        hairstyle = "Short sides, textured top"
    elif face_shape == "Oval":
        hairstyle = "Most styles suit you"
    else:
        hairstyle = "Avoid extra height; add side volume"

    glasses = "Rectangular/square frames" if face_shape in ["Oval", "Long/Oblong"] else "Rounder frames"

    # annotated PIL image (draw a few landmark points)
    annotated = Image.fromarray(cv2.cvtColor(cv_image.copy(), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()

    # draw key points (jaw left/right, nose tip)
    key_points = {
        "Jaw Left": LM["jaw_left"],
        "Jaw Right": LM["jaw_right"],
        "Nose Tip": LM["nose_tip"]
    }
    for name, idx in key_points.items():
        x_px = int(landmarks[idx].x * w)
        y_px = int(landmarks[idx].y * h)
        r = 4
        draw.ellipse((x_px - r, y_px - r, x_px + r, y_px + r), fill=(0, 255, 0))
        draw.text((x_px + 6, y_px - 6), name, fill=(255, 255, 255), font=font)

    report = {
        "face_rating": round(face_rating, 2),
        "face_percentile": face_percentile,
        "attractiveness": round(attractiveness, 2),
        "attractiveness_percentile": attr_percentile,
        "potential_max": round(potential, 2),
        "potential_percentile": potential_percentile,
        "dimorphism": round(dimorphism, 2) if dimorphism is not None else None,
        "dimorphism_label": dim_label,
        "harmony": round(harmony, 2),
        "symmetry": round(symmetry_score, 2),
        "jaw_score": round(jaw_score, 2),
        "cheekbone_score": round(cheekbone_score, 2),
        "eye_spacing_score": round(eye_spacing_score, 2),
        "eye_openness": round(eye_openness, 2),
        "canthal_tilt": round(canthal_score, 2),
        "nose_balance": round(nose_balance, 2),
        "mouth_balance": round(mouth_balance, 2),
        "thirds_score": round(thirds_score, 2),
        "skin_score": round(skin_score, 2),
        "blemish_count": int(blem_count),
        "texture_std": round(texture_std, 2),
        "face_fat": round(face_fat, 2),
        "lip_fullness": round(lip_fullness, 2),
        "face_shape": face_shape,
        "estimated_age": None,  # placeholder to be replaced by ML model if desired
        "confidence": confidence,
        "asymmetry": round(asymmetry_score, 2),
        "suggestions": suggestions,
        "hairstyle_suggestion": hairstyle,
        "glasses_suggestion": glasses,
        "annotated_image": annotated,
        "landmarks": [{"x": float(p.x), "y": float(p.y), "z": float(p.z)} for p in landmarks]
    }

    # simple age estimate based on texture (replace with ML for better result)
    est_age = int(max(16, min(70, 18 + texture_std * 1.5)))
    report["estimated_age"] = f"{est_age} ± 4"

    return report

# -------------------- Export helpers --------------------
def save_report_json(report, path):
    doc = {k: v for k, v in report.items() if k not in ["annotated_image", "landmarks"]}
    doc["landmarks"] = report.get("landmarks", [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

def save_report_pdf(report, path):
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4
    margin = 36
    x = margin
    y = H - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, "Photo Rater Report (1.20 fixed)")
    y -= 22
    c.setFont("Helvetica", 9)
    c.drawString(x, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20

    def line(s):
        nonlocal y
        if y < 100:
            c.showPage()
            y = H - margin
        c.drawString(x, y, s)
        y -= 12

    line(f"Face Rating (structure): {report['face_rating']} / 10 (percentile {report['face_percentile']}%)")
    line(f"Attractiveness: {report['attractiveness']} / 10 (percentile {report['attractiveness_percentile']}%)")
    line(f"Potential (max): {report['potential_max']} / 10 (percentile {report['potential_percentile']}%)")
    if report.get("dimorphism") is not None:
        line(f"{report['dimorphism_label']}: {report['dimorphism']} / 10")
    line(f"Confidence: {report['confidence']}")
    line("")
    line("Structural & Feature Scores:")
    for k in ["jaw_score", "cheekbone_score", "symmetry", "thirds_score", "harmony", "asymmetry"]:
        if k in report:
            line(f" - {k.replace('_',' ').title()}: {report[k]}")
    line("")
    line("Skin & Details:")
    line(f" - Skin score: {report['skin_score']} / 10 (blemishes: {report['blemish_count']})")
    line(f" - Texture STD: {report['texture_std']}")
    line("")
    line("Suggestions:")
    for s in report.get("suggestions", []):
        line(" - " + s)

    # embed annotated image if present
    img = report.get("annotated_image")
    if img:
        bio = io.BytesIO()
        img.thumbnail((360, 360))
        img.save(bio, format="PNG")
        bio.seek(0)
        iw, ih = img.size
        c.drawImage(bio, W - margin - iw, y - ih + 10, width=iw, height=ih)

    c.save()

# -------------------- GUI App --------------------
class PhotoRaterApp:
    def __init__(self, root):
        self.root = root
        root.title("Photo Rater 1.20 (fixed)")
        self.img = None
        self.comp_img = None
        self.report = None

        # controls
        top = tk.Frame(root)
        top.pack(anchor="w", padx=6, pady=6)
        tk.Label(top, text="Biological sex:").pack(side="left")
        self.sex = tk.StringVar(value="M")
        tk.Radiobutton(top, text="Male", variable=self.sex, value="M").pack(side="left")
        tk.Radiobutton(top, text="Female", variable=self.sex, value="F").pack(side="left")

        opts = tk.Frame(root)
        opts.pack(anchor="w", padx=6)
        self.overlay_var = tk.BooleanVar(value=True)
        tk.Checkbutton(opts, text="Show landmark overlay", variable=self.overlay_var).pack(side="left")
        self.compare_var = tk.BooleanVar(value=False)
        tk.Checkbutton(opts, text="Enable comparison mode", variable=self.compare_var).pack(side="left")
        tk.Label(opts, text="Multi-shot N:").pack(side="left", padx=(10, 0))
        self.multi_n = tk.IntVar(value=3)
        tk.Spinbox(opts, from_=1, to=7, width=3, textvariable=self.multi_n).pack(side="left")

        btns = tk.Frame(root)
        btns.pack(anchor="w", padx=6, pady=6)
        tk.Button(btns, text="Upload Photo A", command=self.upload_image).pack(side="left", padx=3)
        tk.Button(btns, text="Upload Photo B (compare)", command=self.upload_compare).pack(side="left", padx=3)
        tk.Button(btns, text="Webcam Capture", command=self.capture_webcam).pack(side="left", padx=3)
        tk.Button(btns, text="Multi-shot Avg (Webcam)", command=self.multi_shot).pack(side="left", padx=3)
        tk.Button(btns, text="Analyze", command=self.analyze).pack(side="left", padx=6)
        tk.Button(btns, text="Save PDF", command=self.save_pdf).pack(side="left", padx=3)
        tk.Button(btns, text="Save JSON", command=self.save_json).pack(side="left", padx=3)
        tk.Button(btns, text="Export Annotated Image", command=self.export_annotated).pack(side="left", padx=3)

        # preview
        self.preview_label = tk.Label(root, text="No image loaded", bg="#111", fg="#fff", width=120, height=20)
        self.preview_label.pack(padx=6, pady=6)

        # text area
        self.text = tk.Text(root, width=120, height=20)
        self.text.pack(padx=6, pady=6)

    # helper: convert and show preview
    def pil_from_cv(self, cv_img, maxsize=(900, 420)):
        pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        pil.thumbnail(maxsize)
        return pil

    def draw_overlay(self, img):
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        out = img.copy()
        if res.multi_face_landmarks:
            mp_draw.draw_landmarks(out, res.multi_face_landmarks[0], mp_face.FACEMESH_TESSELATION,
                                   landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                   connection_drawing_spec=mp_draw.DrawingSpec(color=(80,80,80), thickness=1))
        return out

    def update_preview(self):
        if self.img is None:
            self.preview_label.config(image="", text="No image loaded")
            return
        out = self.img.copy()
        if self.overlay_var.get():
            out = self.draw_overlay(out)
        if self.compare_var.get() and self.comp_img is not None:
            left = out
            right = self.comp_img.copy()
            if self.overlay_var.get():
                right = self.draw_overlay(right)
            h = 420
            left = cv2.resize(left, (int(left.shape[1] * h / left.shape[0]), h))
            right = cv2.resize(right, (int(right.shape[1] * h / right.shape[0]), h))
            concat = np.concatenate([left, right], axis=1)
            pil = Image.fromarray(cv2.cvtColor(concat, cv2.COLOR_BGR2RGB))
            pil.thumbnail((1100, 420))
        else:
            pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            pil.thumbnail((900, 420))
        self.tk_preview = ImageTk.PhotoImage(pil)
        self.preview_label.config(image=self.tk_preview, text="")

    # file ops
    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot open image")
            return
        self.img = img
        self.update_preview()

    def upload_compare(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot open image")
            return
        self.comp_img = img
        self.update_preview()

    def capture_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return
        messagebox.showinfo("Capture", "Press SPACE to capture, ESC to cancel.")
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            cv2.imshow("Webcam - Press SPACE", frame)
            k = cv2.waitKey(1)
            if k == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
            if k == 32:
                cap.release()
                cv2.destroyAllWindows()
                self.img = frame.copy()
                self.update_preview()
                return

    def multi_shot(self):
        n = int(self.multi_n.get())
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam")
            return
        messagebox.showinfo("Multi-shot", f"Will capture {n} shots. Press SPACE for each.")
        shots = []
        for i in range(n):
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.imshow(f"Capture {i+1}/{n} - Press SPACE", frame)
                k = cv2.waitKey(1)
                if k == 27:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if k == 32:
                    shots.append(frame.copy())
                    break
        cap.release()
        cv2.destroyAllWindows()
        avg = np.mean([cv2.cvtColor(s, cv2.COLOR_BGR2RGB).astype(np.float32) for s in shots], axis=0)
        avg_bgr = cv2.cvtColor(avg.astype(np.uint8), cv2.COLOR_RGB2BGR)
        self.img = avg_bgr
        self.update_preview()
        messagebox.showinfo("Multi-shot", "Averaged image created.")

    def analyze(self):
        if self.img is None:
            messagebox.showwarning("No image", "Upload or capture an image first")
            return
        is_male = True if self.sex.get() == "M" else False

        # ML hook placeholder:
        # If you implement a model, preprocess image here and call model.predict()
        # ml_pred = my_model.predict(preprocessed_image)
        # then you may combine/override heuristic fields in report

        report = analyze_single_image(self.img, is_male)
        if "error" in report:
            messagebox.showerror("Error", report["error"])
            return
        self.report = report
        self.show_report(report)

    def show_report(self, rpt):
        self.text.delete(1.0, tk.END)
        lines = []
        lines.append("PHOTO RATER 1.20 — REPORT (fixed)")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append(f"Confidence: {rpt['confidence']}")
        lines.append("")
        lines.append(f"Face Rating (structure): {rpt['face_rating']} / 10 (percentile {rpt['face_percentile']}%)")
        lines.append(f"Attractiveness (perceived): {rpt['attractiveness']} / 10 (percentile {rpt['attractiveness_percentile']}%)")
        lines.append(f"Potential (max achievable): {rpt['potential_max']} / 10 (percentile {rpt['potential_percentile']}%)")
        if rpt.get("dimorphism") is not None:
            lines.append(f"{rpt['dimorphism_label']}: {rpt['dimorphism']} / 10")
        lines.append("")
        lines.append("Structural & feature scores:")
        for k in ["jaw_score", "cheekbone_score", "symmetry", "thirds_score", "harmony", "asymmetry"]:
            if k in rpt:
                lines.append(f" - {k.replace('_',' ').title()}: {rpt[k]}")
        lines.append("")
        lines.append("Features:")
        for k in ["eye_spacing_score", "eye_openness", "canthal_tilt", "nose_balance", "mouth_balance", "lip_fullness", "face_shape", "estimated_age", "face_fat"]:
            if k in rpt:
                lines.append(f" - {k.replace('_',' ').title()}: {rpt[k]}")
        lines.append("")
        lines.append("Skin & details:")
        lines.append(f" - Skin score: {rpt['skin_score']} / 10 (blemishes: {rpt['blemish_count']}, texture std: {rpt['texture_std']})")
        lines.append("")
        lines.append("Suggestions:")
        for s in rpt.get("suggestions", []):
            lines.append(" - " + s)
        lines.append("")
        lines.append("Hair / Glasses advice: " + rpt.get("hairstyle_suggestion", "") + " | " + rpt.get("glasses_suggestion", ""))
        self.text.insert(tk.END, "\n".join(lines))

    def save_pdf(self):
        if not hasattr(self, "report") or self.report is None:
            messagebox.showwarning("No report", "Analyze first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        save_report_pdf(self.report, path)
        messagebox.showinfo("Saved", f"PDF saved to {path}")

    def save_json(self):
        if not hasattr(self, "report") or self.report is None:
            messagebox.showwarning("No report", "Analyze first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        save_report_json(self.report, path)
        messagebox.showinfo("Saved", f"JSON saved to {path}")

    def export_annotated(self):
        if not hasattr(self, "report") or self.report is None:
            messagebox.showwarning("No report", "Analyze first")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not path:
            return
        img = self.report.get("annotated_image")
        if img:
            img.save(path)
            messagebox.showinfo("Saved", f"Annotated image saved to {path}")
        else:
            messagebox.showerror("Error", "No annotated image available")

# -------------------- Launch --------------------
def main():
    root = tk.Tk()
    root.geometry("1200x920")
    app = PhotoRaterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
