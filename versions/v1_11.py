import cv2
import mediapipe as mp
import numpy as np
import math
import os

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

def dist(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

def clamp(x, a=1, b=10):
    return max(a, min(b, x))

def meaning(score):
    if score >= 8: return "Excellent"
    if score >= 7: return "Above average"
    if score >= 6: return "Average"
    if score >= 5: return "Below average"
    return "Weak"

# ================= INPUT CHOICE ================= #

print("Choose input method:")
print("1 - Upload photo")
print("2 - Take webcam photo")
choice = input("Enter choice (1/2): ")

if choice == "1":
    path = input("Enter image path: ").strip('"')
    if not os.path.exists(path):
        print("File not found.")
        exit()
    image = cv2.imread(path)
else:
    cap = cv2.VideoCapture(0)
    print("Align face straight | Press SPACE to capture | ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1)
        if key == 32:
            image = frame.copy()
            break
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()

# ================= FACE ANALYSIS ================= #

rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = face_mesh.process(rgb)

if not result.multi_face_landmarks:
    print("No face detected.")
    exit()

lm = result.multi_face_landmarks[0].landmark

jaw_w = dist(lm[234], lm[454])
face_h = dist(lm[10], lm[152])
face_w = jaw_w
eye_d = dist(lm[33], lm[263])

fwhr = face_w / face_h
eye_ratio = eye_d / face_w

jaw = clamp(jaw_w / face_h * 12)
sym = clamp(10 - abs(lm[33].x - (1 - lm[263].x)) * 120)
zyg = clamp(dist(lm[127], lm[356]) / face_w * 12)
eye_score = clamp(10 - abs(eye_ratio - 0.46) * 20)
canthal = clamp(((lm[33].y - lm[263].y) * -100 + 4) * 1.5)

skin = clamp(10 - np.std(image) / 25)
beard = clamp(jaw + 1.5)
face_fat = clamp(10 - fwhr * 8)

normal = clamp((jaw + sym + eye_score + skin) / 4)
psl = clamp((jaw*1.4 + sym*1.3 + zyg + canthal) / 5)

# ================= TYPES ================= #

face_type = "Oval"
if fwhr > 0.93: face_type = "Round"
elif fwhr < 0.78: face_type = "Long"
elif jaw > 7.5: face_type = "Square"

eye_type = "Normal"
if eye_ratio > 0.48: eye_type = "Wide-set"
elif eye_ratio < 0.42: eye_type = "Close-set"

eye_area = "Neutral"
if lm[159].y - lm[145].y < 0.01:
    eye_area = "Hooded"

# ================= DETAILED REPORT ================= #

print("\n=========== DETAILED FACE REPORT ===========\n")

print(f"Normal Rating: {normal:.1f}/10 ({meaning(normal)})")
print(f"PSL Rating: {psl:.1f}/10 ({meaning(psl)})\n")

def section(name, score, desc, opt):
    print(f"{name}: {score:.1f}/10 ({meaning(score)})")
    print(f" - Interpretation: {desc}")
    print(f" - Optimization: {opt}\n")

section(
    "Jawline",
    jaw,
    "Mandibular width and chin projection",
    "Light beard or stubble improves definition"
)

section(
    "Facial Symmetry",
    sym,
    "Left-right proportional balance",
    "Hairstyles avoiding middle part help"
)

section(
    "Cheekbones (Zygos)",
    zyg,
    "Mid-face width and prominence",
    "Leaner body fat enhances cheekbone visibility"
)

section(
    "Eye Area",
    eye_score,
    f"{eye_type} eyes with {eye_area.lower()} lids",
    "Proper sleep and under-eye care help appearance"
)

print("STRUCTURAL DATA")
print(f"Face Type: {face_type}")
print(f"FWHR: {fwhr:.2f}")
print(f"Canthal Tilt Score: {canthal:.1f}")
print(f"Skin Rating: {skin:.1f}")
print(f"Beard Potential: {beard:.1f}")
print(f"Face Fat Level: {face_fat:.1f}\n")

print("POSITIVES")
if jaw > 7: print("- Strong jaw structure")
if zyg > 7: print("- Prominent cheekbones")
if canthal > 7: print("- Positive eye tilt")

print("\nNEGATIVES")
if sym < 6.5: print("- Noticeable asymmetry")
if face_fat > 6.5: print("- Facial fat softens angles")
if eye_score < 6.5: print("- Eye proportional imbalance")

print("\n===========================================\n")
