import cv2
import numpy as np
import csv
import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_ASSET_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
CSV_FILE = os.path.join(PROJECT_ROOT, "data", "static_asl_data.csv")

class_counts = {chr(i).upper(): 0 for i in range(97, 123)}
if os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header
        for row in reader:
            if row and row[0] in class_counts:
                class_counts[row[0]] += 1

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm Base Closure
]

latest_landmarks = None
current_frame_features = None

def tracking_callback(result: "vision.HandLandmarkerResult", output_image: mp.Image, timestamp_ms: int): # type: ignore
    global latest_landmarks, current_frame_features
    
    if (result and hasattr(result, 'hand_landmarks') and result.hand_landmarks and 
                   hasattr(result, 'hand_world_landmarks') and result.hand_world_landmarks):
        
        latest_landmarks = result.hand_landmarks[0]
        
        world_landmarks = result.hand_world_landmarks[0]
        world_wrist = world_landmarks[0]
        
        features = []
        for lm in world_landmarks:
            features.extend([lm.x - world_wrist.x, lm.y - world_wrist.y, lm.z - world_wrist.z])
        current_frame_features = features
    else:
        latest_landmarks = None
        current_frame_features = None

base_options = python.BaseOptions(model_asset_path=MODEL_ASSET_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=tracking_callback,
    min_hand_detection_confidence=0.35,
    min_hand_presence_confidence=0.35,
    min_tracking_confidence=0.35
)
detector = vision.HandLandmarker.create_from_options(options)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f'coord_{i}' for i in range(63)]
        writer.writerow(header)

cap = cv2.VideoCapture(0)
print("=== ANGLE-INVARIANT STATIC ASL DATA COLLECTOR ===")
time.sleep(1.0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: continue
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    channels = list(cv2.split(yuv))
    channels[0] = cv2.equalizeHist(channels[0])
    yuv = cv2.merge(channels)
    enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    timestamp_ms = int(time.time() * 1000)
    image_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detector.detect_async(mp_image, timestamp_ms)
    
    local_landmarks = latest_landmarks
    local_features = current_frame_features
    
    if local_landmarks is not None and local_features is not None:
        for connection in HAND_CONNECTIONS:
            start_point = (int(local_landmarks[connection[0]].x * w), int(local_landmarks[connection[0]].y * h))
            end_point = (int(local_landmarks[connection[1]].x * w), int(local_landmarks[connection[1]].y * h))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(frame, start_point, end_point, (0, 215, 255), 1, cv2.LINE_AA)
            
        for lm in local_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 50, 255), -1, cv2.LINE_AA)

    cv2.imshow("Static Collection - Hold Sign and Press Key", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:
        break
    elif 97 <= key <= 122:
        letter = chr(key).upper()
        if letter in ['J', 'Z']:
            print("[SKIP] J and Z must be recorded using collect_motion_data.py!")
            continue
            
        if local_features:
            with open(CSV_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([letter] + local_features)  # Logs angle-invariant metric data arrays
            
            class_counts[letter] += 1
            print(f"[RECORDED] Saved sample #{class_counts[letter]} for letter: {letter}")
        else:
            print("[WARNING] No hand tracked! Adjust position.")

cap.release()
cv2.destroyAllWindows()