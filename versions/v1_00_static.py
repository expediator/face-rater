import cv2
import mediapipe as mp
import numpy as np
import math

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

cap = cv2.VideoCapture(0)

print("Align face straight | Press SPACE to analyze | ESC to exit")

frame_captured = False
saved_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if not frame_captured:
        cv2.putText(frame, "Press SPACE to ANALYZE", (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Looksmax Analyzer", frame)

    key = cv2.waitKey(1)
    if key == 32:  # SPACE
        frame_captured = True
        saved_frame = frame.copy()
        break
    if key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# ---------------- ANALYSIS ---------------- #

rgb = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)
result = face_mesh.process(rgb)

if not result.multi_face_landmarks:
    print("No face detected.")
    exit()

face = result.multi_face_landmarks[0]
lm = face.landmark

# ---------------- BASIC MEASUREMENTS ---------------- #
jaw_w = dist(lm[234], lm[454])
face_h = dist(lm[10], lm[152])
face_w = jaw_w
eye_d = dist(lm[33], lm[263])

fwhr = face_w / face_h

# ---------------- SCORES ---------------- #
jaw_score = clamp(jaw_w / face_h * 12)
symmetry = clamp(10 - abs(lm[33].x - (1 - lm[263].x)) * 120)
zyg = clamp(dist(lm[127], lm[356]) / face_w * 12)

eye_spacing_ratio = eye_d / face_w
eye_score = clamp(10 - abs(eye_spacing_ratio - 0.46) * 20)

canthal = (lm[33].y - lm[263].y) * -100
canthal_score = clamp((canthal + 4) * 1.5)

# ---------------- TYPES ---------------- #
face_type = "Oval"
if fwhr > 0.93:
    face_type = "Round"
elif fwhr < 0.78:
    face_type = "Long"
elif jaw_score > 7.5:
    face_type = "Square"

eye_type = "Normal"
if eye_spacing_ratio > 0.48:
    eye_type = "Wide-set"
elif eye_spacing_ratio < 0.42:
    eye_type = "Close-set"

eye_area = "Neutral"
if lm[159].y - lm[145].y < 0.01:
    eye_area = "Hooded"

# ---------------- SKIN / BEARD / FAT (HEURISTIC) ---------------- #
skin = clamp(10 - np.std(saved_frame)/25)
beard = clamp(jaw_score + 1.5)
face_fat = clamp(10 - fwhr*8)

# ---------------- FINAL RATINGS ---------------- #
normal_rating = clamp((jaw_score + symmetry + eye_score + skin) / 4)
psl_rating = clamp((jaw_score*1.4 + symmetry*1.3 + zyg + canthal_score) / 5)

# ---------------- REPORT ---------------- #
print("\n========== FACE ANALYSIS REPORT ==========")
print(f"Normal Rating: {normal_rating:.1f}/10")
print(f"PSL Rating: {psl_rating:.1f}/10\n")

print("STRUCTURE")
print(f"Jawline: {jaw_score:.1f}")
print(f"Zygos: {zyg:.1f}")
print(f"Symmetry: {symmetry:.1f}")
print(f"FWHR: {fwhr:.2f}")

print("\nEYES")
print(f"Eye score: {eye_score:.1f}")
print(f"Eye spacing: {eye_type}")
print(f"Canthal tilt: {canthal_score:.1f}")
print(f"Eye area: {eye_area}")

print("\nOTHER")
print(f"Face type: {face_type}")
print(f"Skin rating: {skin:.1f}")
print(f"Beard potential: {beard:.1f}")
print(f"Face fat level: {face_fat:.1f}")

# ---------------- POSITIVES & NEGATIVES ---------------- #
print("\nPOSITIVES")
if jaw_score > 7:
    print("- Strong jaw structure")
if zyg > 7:
    print("- Prominent cheekbones")
if canthal_score > 7:
    print("- Positive canthal tilt")

print("\nNEGATIVES")
if symmetry < 6.5:
    print("- Facial asymmetry noticeable")
if face_fat > 6.5:
    print("- Fat softens facial angles")
if eye_score < 6.5:
    print("- Eye proportion imbalance")

print("\n==========================================")
