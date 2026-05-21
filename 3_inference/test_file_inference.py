import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import filedialog, messagebox

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

STATIC_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "static_model.h5")
STATIC_CLASSES_PATH = os.path.join(PROJECT_ROOT, "models", "static_classes.npy")
MOTION_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "motion_model.h5")
LANDMARKER_TASK_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

static_model = tf.keras.models.load_model(STATIC_MODEL_PATH)
static_classes = np.load(STATIC_CLASSES_PATH, allow_pickle=True)
motion_model = tf.keras.models.load_model(MOTION_MODEL_PATH)
motion_classes = {0: "J", 1: "Z"}

base_options = python.BaseOptions(model_asset_path=LANDMARKER_TASK_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def scale_frame(frame, max_width=960, max_height=720):
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h)
    
    if abs(scale - 1.0) > 0.01:
        new_w = int(w * scale)
        new_h = int(h * scale)
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
        
    return frame

def extract_landmarks(frame):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)
    
    if (result and hasattr(result, 'hand_landmarks') and result.hand_landmarks and 
                   hasattr(result, 'hand_world_landmarks') and result.hand_world_landmarks):
        
        local_landmarks = result.hand_landmarks[0]
        world_landmarks = result.hand_world_landmarks[0]
        world_wrist = world_landmarks[0]
        
        features = []
        for lm in world_landmarks:
            features.extend([lm.x - world_wrist.x, lm.y - world_wrist.y, lm.z - world_wrist.z])
            
        return features, local_landmarks
    return None, None

def draw_skeleton(frame, local_landmarks):
    h, w, _ = frame.shape
    for connection in HAND_CONNECTIONS:
        start_point = (int(local_landmarks[connection[0]].x * w), int(local_landmarks[connection[0]].y * h))
        end_point = (int(local_landmarks[connection[1]].x * w), int(local_landmarks[connection[1]].y * h))
        cv2.line(frame, start_point, end_point, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(frame, start_point, end_point, (0, 215, 255), 1, cv2.LINE_AA)
        
    for lm in local_landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 4, (0, 50, 255), -1, cv2.LINE_AA)

def process_image(file_path):
    raw_frame = cv2.imread(file_path)
    if raw_frame is None:
        messagebox.showerror("Error", f"Could not read image: {file_path}")
        return

    frame = scale_frame(raw_frame)
    features, local_landmarks = extract_landmarks(frame)
    
    if features:
        draw_skeleton(frame, local_landmarks)
        input_data = np.array(features).reshape(1, -1)
        pred = static_model.predict(input_data, verbose=0)
        class_idx = np.argmax(pred)
        letter = static_classes[class_idx]
        confidence = pred[0][class_idx] * 100
        
        text = f"Predicted: {letter} ({confidence:.1f}%)"
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "No Hand Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    out_path = os.path.join(SCRIPT_DIR, "output_" + os.path.basename(file_path))
    cv2.imwrite(out_path, frame)
    
    cv2.namedWindow("Image Test Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Test Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Could not open video: {file_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        messagebox.showerror("Error", "Empty video file selected.")
        return
        
    sample_scaled = scale_frame(first_frame)
    h, w, _ = sample_scaled.shape
    
    out_path = os.path.join(SCRIPT_DIR, "output_" + os.path.basename(file_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    sequence_buffer = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    cv2.namedWindow("Video Test Result", cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, raw_frame = cap.read()
        if not ret:
            break

        frame = scale_frame(raw_frame)
        features, local_landmarks = extract_landmarks(frame)
        display_text = "Tracking..."

        if features:
            draw_skeleton(frame, local_landmarks)

            static_input = np.array(features).reshape(1, -1)
            static_pred = static_model.predict(static_input, verbose=0)
            static_letter = static_classes[np.argmax(static_pred)]
            display_text = f"Static: {static_letter}"

            sequence_buffer.append(features)
            if len(sequence_buffer) > 30:
                sequence_buffer.pop(0)

            if len(sequence_buffer) == 30:
                motion_input = np.array(sequence_buffer).reshape(1, 30, 63)
                motion_pred = motion_model.predict(motion_input, verbose=0)
                motion_idx = np.argmax(motion_pred)
                motion_conf = motion_pred[0][motion_idx]

                if motion_conf > 0.85:
                    motion_letter = motion_classes[motion_idx]
                    display_text = f"Motion: {motion_letter} ({motion_conf*100:.1f}%)"
        else:
            sequence_buffer.clear()
            display_text = "No Hand Detected"

        cv2.putText(frame, display_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)
        
        cv2.imshow("Video Test Result", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Processed video saved to:\n{out_path}")

def browse_file():
    file_types = [
        ("All Supported Files", "*.jpg *.jpeg *.png *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.wmv"),
        ("Images", "*.jpg *.jpeg *.png *.bmp *.webp"),
        ("Videos", "*.mp4 *.avi *.mov *.mkv *.wmv")
    ]
    target_path = filedialog.askopenfilename(title="Select ASL Image or Video", filetypes=file_types)
    
    if not target_path:
        return
        
    file_extension = os.path.splitext(target_path)[1].lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        process_image(target_path)
    elif file_extension in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
        process_video(target_path)
    else:
        messagebox.showerror("Unsupported Format", "Selected file extension is not supported.")

def setup_gui():
    root = tk.Tk()
    root.title("ASL Inference Testing Hub")
    root.geometry("450x200")
    root.resizable(False, False)
    
    label_title = tk.Label(root, text="ASL Classifier Model Evaluator", font=("Helvetica", 14, "bold"), pady=10)
    label_title.pack()
    
    label_desc = tk.Label(root, text="Select a recorded photo or video file to analyze hand gestures.\nFrames dynamically auto-scale to ensure complete visibility.", font=("Helvetica", 10), fg="gray")
    label_desc.pack(pady=5)
    
    btn_browse = tk.Button(root, text="Insert Image / Video", font=("Helvetica", 11, "bold"), bg="#00d7ff", fg="black", padx=20, pady=10, command=browse_file)
    btn_browse.pack(pady=15)
    
    root.mainloop()

if __name__ == "__main__":
    setup_gui()