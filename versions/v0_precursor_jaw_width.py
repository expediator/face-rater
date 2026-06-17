import cv2
import mediapipe as mp
import numpy as np
import math

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False)

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

cap = cv2.VideoCapture(0)

print("Press SPACE to analyze face | ESC to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Jawline points
            left_jaw = face.landmark[234]
            right_jaw = face.landmark[454]

            # Face height
            forehead = face.landmark[10]
            chin = face.landmark[152]

            jaw_width = distance(left_jaw, right_jaw)
            face_height = distance(forehead, chin)

            ratio = jaw_width / face_height

            jaw_score = min(10, max(4, ratio * 12))

            cv2.putText(
                frame,
                f"Jaw Score: {jaw_score:.1f}/10",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow("Looksmax Analyzer", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
