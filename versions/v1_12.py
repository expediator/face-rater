import cv2
import mediapipe as mp
import numpy as np
import math
import os

# ===================== SETUP =====================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=True)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def clamp(x, minv=1, maxv=10):
    return max(minv, min(maxv, x))

def meaning(score):
    if score >= 8: return "Excellent"
    if score >= 6.5: return "Good"
    if score >= 5: return "Average"
    return "Below Average"

# ===================== SEX SELECTION =====================
print("\nSelect biological sex for analysis:")
print("1 - Male")
print("2 - Female")
sex_choice = input("Enter choice (1/2): ").strip()

if sex_choice not in ["1", "2"]:
    print("Invalid choice.")
    exit()

is_male = sex_choice == "1"

# ===================== INPUT METHOD =====================
print("\nChoose input method:")
print("1 - Upload photo")
print("2 - Take webcam photo")
method = input("Enter choice (1/2): ").strip()

if method == "1":
    img_path = input("Enter FULL image path: ").strip().strip('"')
    if not os.path.isfile(img_path):
        print("Invalid image path.")
        exit()
    image = cv2.imread(img_path)

elif method == "2":
    cap = cv2.VideoCapture(0)
    print("Press SPACE to capture image | ESC to exit")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture", frame)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()
        if key == 32:
            image = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    print("Invalid choice.")
    exit()

# ===================== FACE ANALYSIS =====================
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb)

if not results.multi_face_landmarks:
    print("No face detected.")
    exit()

face = results.multi_face_landmarks[0]

# ===================== LANDMARKS =====================
left_jaw = face.landmark[234]
right_jaw = face.landmark[454]
chin = face.landmark[152]
forehead = face.landmark[10]

left_eye_outer = face.landmark[33]
right_eye_outer = face.landmark[263]
left_eye_inner = face.landmark[133]
right_eye_inner = face.landmark[362]

left_cheek = face.landmark[93]
right_cheek = face.landmark[323]

# ===================== MEASUREMENTS =====================
jaw_width = distance(left_jaw, right_jaw)
face_height = distance(forehead, chin)
face_width = distance(left_cheek, right_cheek)
eye_width = distance(left_eye_inner, left_eye_outer)

fwhr = face_width / face_height
jaw_ratio = jaw_width / face_height

# ===================== SCORES =====================
jaw_score = clamp(jaw_ratio * 12)
symmetry_score = clamp((1 - abs((left_eye_outer.y - right_eye_outer.y))) * 10)
eye_score = clamp(eye_width * 40)
fwhr_score = clamp(fwhr * 5)
canthal_tilt = clamp((right_eye_outer.y - left_eye_outer.y) * 25 + 5)

normal_scale = clamp((jaw_score + symmetry_score + eye_score + fwhr_score) / 4)
psl_scale = clamp(normal_scale - 0.5)

# ===================== GENDER TRAITS =====================
if is_male:
    masculinity = clamp(
        (jaw_score * 1.4 + fwhr_score + canthal_tilt) / 3
    )
else:
    femininity = clamp(
        (eye_score * 1.4 + symmetry_score + (10 - jaw_score)) / 3
    )

# ===================== REPORT =====================
print("\n================ FACE ANALYSIS REPORT ================\n")

print(f"Jawline: {jaw_score:.1f}/10 ({meaning(jaw_score)})")
print(f"Symmetry: {symmetry_score:.1f}/10 ({meaning(symmetry_score)})")
print(f"Eye Area: {eye_score:.1f}/10 ({meaning(eye_score)})")
print(f"Canthal Tilt: {canthal_tilt:.1f}/10")
print(f"FWHR: {fwhr_score:.1f}/10")

if is_male:
    print(f"Masculinity: {masculinity:.1f}/10 ({meaning(masculinity)})")
else:
    print(f"Femininity: {femininity:.1f}/10 ({meaning(femininity)})")

print(f"\nNormal Scale Rating: {normal_scale:.1f}/10")
print(f"PSL Scale Rating: {psl_scale:.1f}/10")

# ===================== POSITIVES / NEGATIVES =====================
print("\nPositives:")
if jaw_score > 7: print("- Strong jaw structure")
if symmetry_score > 7: print("- Good facial symmetry")
if eye_score > 7: print("- Attractive eye area")
if is_male and masculinity > 7: print("- Strong masculine traits")
if not is_male and femininity > 7: print("- Soft and feminine facial harmony")

print("\nNegatives:")
if jaw_score < 5: print("- Jaw lacks definition")
if symmetry_score < 5: print("- Facial asymmetry present")
if eye_score < 5: print("- Eye area could be improved")
if is_male and jaw_score < 6: print("- Masculine lower third could be stronger")
if not is_male and jaw_score > 7.5: print("- Jaw slightly strong for feminine ideal")

print("\n======================================================")
