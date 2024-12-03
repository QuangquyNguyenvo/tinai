import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import mediapipe as mp
import math
import time
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

def duongthang(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    angle = math.atan2(dy, dx) * 180 / math.pi
    return abs(angle)

def khoangcach(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(dx * dx + dy * dy)

def check_fall(landmarks):
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
    mid_ankle_x = (left_ankle.x + right_ankle.x) / 2
    mid_ankle_y = (left_ankle.y + right_ankle.y) / 2
    angle = duongthang(nose, type('Landmark', (object,), {'x': mid_ankle_x, 'y': mid_ankle_y}) )
    dis = khoangcach(nose, type('Landmark', (object,), {'x': mid_ankle_x, 'y': mid_ankle_y}))
    if angle < 30 or dis<0.3:
        return True
    return False

cap = cv2.VideoCapture(0)
prev_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        mid_ankle_x = (left_ankle.x + right_ankle.x) / 2
        mid_ankle_y = (left_ankle.y + right_ankle.y) / 2
        cv2.line(frame, (int(nose.x * frame.shape[1]), int(nose.y * frame.shape[0])), 
                 (int(mid_ankle_x * frame.shape[1]), int(mid_ankle_y * frame.shape[0])), (0, 255, 0), 2)
        if check_fall(results.pose_landmarks.landmark):
            cv2.putText(frame, "Fall detected!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Fall Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
