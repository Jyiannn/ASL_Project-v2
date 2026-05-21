import os
import cv2
import numpy as np
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

MODEL_ASSET_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")
DATA_FILE = os.path.join(PROJECT_ROOT, "data", "motion_asl_data.npy")
SEQUENCE_LENGTH = 30

label_map = {'J': 0, 'Z': 1}
class_counts = {'J': 0, 'Z': 0}

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'rb') as f:
        _ = np.load(f)
        y_existing = np.load(f)
        for label_val in y_existing:
            if label_val == 0: class_counts['J'] += 1
            elif label_val == 1: class_counts['Z'] += 1

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

global_landmarks = None
global_features = None
global_world_wrist = None

def tracking_callback(result: "vision.HandLandmarkerResult", output_image: mp.Image, timestamp_ms: int):
    global global_landmarks, global_features, global_world_wrist
    
    if (result and hasattr(result, 'hand_landmarks') and result.hand_landmarks and 
                   hasattr(result, 'hand_world_landmarks') and result.hand_world_landmarks):
        
        global_landmarks = result.hand_landmarks[0]
        
        world_landmarks = result.hand_world_landmarks[0]
        world_wrist = world_landmarks[0]
        
        global_world_wrist = [world_wrist.x, world_wrist.y, world_wrist.z]
        
        features = []
        for lm in world_landmarks:
            features.extend([lm.x - world_wrist.x, lm.y - world_wrist.y, lm.z - world_wrist.z])
        global_features = features
    else:
        global_landmarks = None
        global_features = None
        global_world_wrist = None

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

collected_sequences = []
collected_labels = []

cap = cv2.VideoCapture(0)
print("=== ANGLE-INVARIANT & TRAJECTORY-AWARE LSTM MOTION DATA COLLECTOR ===")
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
    
    local_landmarks = global_landmarks
    local_features = global_features
    
    if local_landmarks is not None and local_features is not None:
        for connection in HAND_CONNECTIONS:
            start_point = (int(local_landmarks[connection[0]].x * w), int(local_landmarks[connection[0]].y * h))
            end_point = (int(local_landmarks[connection[1]].x * w), int(local_landmarks[connection[1]].y * h))
            cv2.line(frame, start_point, end_point, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.line(frame, start_point, end_point, (0, 215, 255), 1, cv2.LINE_AA)
        for lm in local_landmarks:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 50, 255), -1, cv2.LINE_AA)

    ui_text = f"Total Database Size: J [{class_counts['J']}]  Z [{class_counts['Z']}]"
    cv2.putText(frame, ui_text, (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("LSTM Collector - Press J or Z", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:
        break
    elif key in [ord('j'), ord('z')]:
        target_letter = chr(key).upper()
        class_counts[target_letter] += 1  
        
        sequence_buffer = []
        sequence_start_wrist = None
        print(f"[START] Recording motion path for '{target_letter}'... Make sure to draw the gesture path!")
        
        while len(sequence_buffer) < SEQUENCE_LENGTH:
            success, frame = cap.read()
            if not success: continue
            frame = cv2.flip(frame, 1)
            
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            channels = list(cv2.split(yuv))
            channels[0] = cv2.equalizeHist(channels[0])
            yuv = cv2.merge(channels)
            enhanced_frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            
            timestamp_ms = int(time.time() * 1000)
            image_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            detector.detect_async(mp_image, timestamp_ms)
            
            local_landmarks = global_landmarks
            local_features = global_features
            local_wrist = global_world_wrist
            
            if local_landmarks is not None and local_features is not None and local_wrist is not None:
                if sequence_start_wrist is None:
                    sequence_start_wrist = local_wrist
                
                trajectory = [
                    local_wrist[0] - sequence_start_wrist[0],
                    local_wrist[1] - sequence_start_wrist[1],
                    local_wrist[2] - sequence_start_wrist[2]
                ]
                
                combined_features = local_features + trajectory
                sequence_buffer.append(combined_features)
                
                for connection in HAND_CONNECTIONS:
                    start_point = (int(local_landmarks[connection[0]].x * w), int(local_landmarks[connection[0]].y * h))
                    end_point = (int(local_landmarks[connection[1]].x * w), int(local_landmarks[connection[1]].y * h))
                    cv2.line(frame, start_point, end_point, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.line(frame, start_point, end_point, (0, 215, 255), 1, cv2.LINE_AA)
                for lm in local_landmarks:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 50, 255), -1, cv2.LINE_AA)
            
            cv2.putText(frame, f"Recording {target_letter} #{class_counts[target_letter]}: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("LSTM Collector - Press J or Z", frame)
            cv2.waitKey(25)
            
        collected_sequences.append(sequence_buffer)
        collected_labels.append(label_map[target_letter])
        print(f"[SUCCESS] Saved sequence #{class_counts[target_letter]} for '{target_letter}'")

cap.release()
cv2.destroyAllWindows()

if len(collected_sequences) > 0:
    X_new = np.array(collected_sequences)
    y_new = np.array(collected_labels)
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            X_old = np.load(f)
            y_old = np.load(f)
        X_final = np.vstack((X_old, X_new))
        y_final = np.concatenate((y_old, y_new))
    else:
        X_final, y_final = X_new, y_new
        
    with open(DATA_FILE, 'wb') as f:
        np.save(f, X_final)
        np.save(f, y_final)
    print(f"\n[FINISHED] Total dataset file size updated: {X_final.shape}")