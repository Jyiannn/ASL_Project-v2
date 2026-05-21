import os
import cv2
import numpy as np
import tensorflow as tf
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "models"))

STATIC_MODEL_PATH = os.path.join(MODELS_DIR, "static_model.h5")
STATIC_CLASSES_PATH = os.path.join(MODELS_DIR, "static_classes.npy")
LANDMARKER_TASK_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

try:
    static_model = tf.keras.models.load_model(STATIC_MODEL_PATH)
    static_classes = np.load(STATIC_CLASSES_PATH, allow_pickle=True)
except FileNotFoundError as e:
    raise FileNotFoundError(f"Missing static model files inside 'models/'! Ensure static network is trained: {e}")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

latest_landmarks = None
latest_world_landmarks = None

def tracking_callback(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_landmarks, latest_world_landmarks
    if result.hand_landmarks and result.hand_world_landmarks:
        latest_landmarks = result.hand_landmarks[0]
        latest_world_landmarks = result.hand_world_landmarks[0]
    else:
        latest_landmarks = None
        latest_world_landmarks = None

current_word = ""
hand_seen_time = None
last_raw_sign = ""
no_hand_time = None

HOLD_DURATION = 2.0
RESET_DURATION = 3.0

base_options = python.BaseOptions(model_asset_path=LANDMARKER_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=tracking_callback
)

cap = cv2.VideoCapture(0)
start_time = time.time()

with vision.HandLandmarker.create_from_options(options) as landmarker:
    print("[INFO] Static Inference Engine Running... Press ESC to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        local_landmarks = latest_landmarks
        local_world_landmarks = latest_world_landmarks
        
        raw_detected_sign = ""
        static_conf = 0.0
        
        if local_landmarks is not None and local_world_landmarks is not None:
            no_hand_time = None
            
            for connection in HAND_CONNECTIONS:
                start_point = (int(local_landmarks[connection[0]].x * w), int(local_landmarks[connection[0]].y * h))
                end_point = (int(local_landmarks[connection[1]].x * w), int(local_landmarks[connection[1]].y * h))
                cv2.line(frame, start_point, end_point, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.line(frame, start_point, end_point, (0, 215, 255), 1, cv2.LINE_AA)
            for lm in local_landmarks:
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 50, 255), -1, cv2.LINE_AA)
                
            wrist = local_world_landmarks[0]
            local_features = []
            for lm in local_world_landmarks:
                local_features.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
                
            fnn_input_batch = np.expand_dims(np.array(local_features), axis=0)
            static_preds = static_model.predict(fnn_input_batch, verbose=0)[0]
            static_idx = np.argmax(static_preds)
            static_conf = static_preds[static_idx]
            
            raw_detected_sign = str(static_classes[static_idx]).upper()
            
            text_output = f"Letter: {raw_detected_sign} ({static_conf*100:.1f}%)"
            cv2.putText(frame, text_output, (22, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text_output, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2, cv2.LINE_AA)
            
            if static_conf > 0.70:
                if raw_detected_sign == last_raw_sign:
                    if hand_seen_time is None:
                        hand_seen_time = time.time()
                    else:
                        elapsed = time.time() - hand_seen_time
                        if elapsed >= HOLD_DURATION:
                            current_word += raw_detected_sign
                            hand_seen_time = None
                            last_raw_sign = ""
                else:
                    last_raw_sign = raw_detected_sign
                    hand_seen_time = time.time()
            else:
                hand_seen_time = None
                last_raw_sign = ""
        else:
            cv2.putText(frame, "No Hand Tracked", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
            hand_seen_time = None
            last_raw_sign = ""
            
            if no_hand_time is None:
                no_hand_time = time.time()
            elif (time.time() - no_hand_time) >= RESET_DURATION:
                current_word = ""
                no_hand_time = None
                
        cv2.rectangle(frame, (0, 0), (w, 60), (15, 15, 15), -1)
        word_display = f"Word: {current_word}"
        cv2.putText(frame, word_display, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2, cv2.LINE_AA)
        
        if hand_seen_time and last_raw_sign != "":
            elapsed = time.time() - hand_seen_time
            progress_ratio = min(elapsed / HOLD_DURATION, 1.0)
            cv2.rectangle(frame, (0, 55), (int(w * progress_ratio), 60), (0, 215, 255), -1)
            
        cv2.imshow("Static Inference", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()