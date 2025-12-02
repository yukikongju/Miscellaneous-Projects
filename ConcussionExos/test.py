import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            h, w, c = frame.shape

            # Iris indices (mediapipe uses these)
            left_iris = [474, 475, 476, 477]
            right_iris = [469, 470, 471, 472]

            def get_center(index_list):
                pts = []
                for idx in index_list:
                    x = int(face.landmark[idx].x * w)
                    y = int(face.landmark[idx].y * h)
                    pts.append((x, y))
                pts = np.array(pts)
                return np.mean(pts, axis=0).astype(int)

            left_center = get_center(left_iris)
            right_center = get_center(right_iris)

            cv2.circle(frame, tuple(left_center), 3, (0, 255, 0), -1)
            cv2.circle(frame, tuple(right_center), 3, (0, 255, 0), -1)

            # Rough gaze direction (horizontal only)
            # Pick 2 eyelid landmarks
            left_eye_left = np.array([face.landmark[33].x * w, face.landmark[33].y * h])
            left_eye_right = np.array(
                [face.landmark[133].x * w, face.landmark[133].y * h]
            )

            eye_width = np.linalg.norm(left_eye_right - left_eye_left)
            pos = (left_center[0] - left_eye_left[0]) / eye_width

            if pos < 0.4:
                gaze = "LEFT"
            elif pos > 0.6:
                gaze = "RIGHT"
            else:
                gaze = "CENTER"

            cv2.putText(
                frame,
                f"Gaze: {gaze}",
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    cv2.imshow("Eye Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
