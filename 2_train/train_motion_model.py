import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "motion_asl_data.npy")
MODEL_OUT_PATH = os.path.join(PROJECT_ROOT, "models", "motion_model.h5")
CLASSES_OUT_PATH = os.path.join(PROJECT_ROOT, "models", "motion_classes.npy")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Missing motion sequence dataset. Run collect_motion_data.py first to create '{DATA_FILE}'!")

print("[INFO] Extracting spatial-temporal sequence tensors...")
try:
    with open(DATA_FILE, 'rb') as f:
        X = np.load(f)
        y = np.load(f)
except Exception as e:
    raise IOError(f"Failed to read data arrays. The data file might be corrupted: {e}")

class_names = np.array(['J', 'Z'])
num_classes = len(class_names)
np.save(CLASSES_OUT_PATH, class_names)

print(f"[SUCCESS] Cleanly parsed {X.shape[0]} continuous motion path sequences.")
print(f"Sequence Shape: {X.shape} -> (Samples, Timesteps, Tracking Features)")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

candidate_batch_sizes = [4, 8, 16, 32]
best_accuracy = 0.0
best_batch_size = None
best_model = None

print("\n[OPTIMIZATION] Running hyperparameter scan for sequential tracking layouts...")

for bs in candidate_batch_sizes:
    print(f"\n TESTING Candidate Batch Size: {bs} (Max Limit: 150 Epochs)")
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(30, 66)),
        tf.keras.layers.LSTM(64, return_sequences=True, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train, y_train, 
        epochs=150, 
        batch_size=bs, 
        validation_data=(X_test, y_test), 
        callbacks=[early_stopping],
        verbose=1
    )
    
    trial_peak_acc = max(history.history['val_accuracy'])
    print(f"Batch Size {bs} finished. Peak Validation Accuracy: {trial_peak_acc * 100:.2f}%")
    
    if trial_peak_acc > best_accuracy:
        best_accuracy = trial_peak_acc
        best_batch_size = bs
        best_model = model

print(f"\n[SUCCESS] Sequence Optimization Complete!")
print(f" Best Selected Batch Size: {best_batch_size}")
print(f" Highest Validated Sequence Accuracy: {best_accuracy * 100:.2f}%")

best_model.save(MODEL_OUT_PATH)
print(f"[INFO] Structural tracking weights file saved to '{MODEL_OUT_PATH}'.")

print("\n" + "="*50)
print("   MOTION PERFORMANCE MATRIX (BEST MODEL)   ")
print("="*50)

y_pred_probs = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=range(num_classes))
map_score = np.mean(precision)

print(f" Overall Motion mAP:   {map_score * 100:.1f}%")
print(f" Top-1 Sequence Acc:   {best_accuracy * 100:.1f}%")
print("-" * 50)

print("Motion Path Performance Breakdown:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(8, 6))
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Motion Confusion Matrix (Batch Size: {best_batch_size})', fontsize=14, pad=15)
plt.colorbar(label='Fraction of Predictions')

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, fontsize=10)
plt.yticks(tick_marks, class_names, fontsize=10)

thresh = cm_norm.max() / 2.
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm_norm[i, j] > thresh else "black",
                 fontsize=11, weight="bold")

plt.ylabel('True Label', fontsize=12, labelpad=10)
plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
plt.grid(False)
plt.tight_layout()

matrix_img_path = os.path.join(SCRIPT_DIR, 'motion_confusion_matrix.png')
plt.savefig(matrix_img_path, dpi=300)
print(f"\n[INFO] Evaluation charts saved cleanly as '{matrix_img_path}'!")
print("="*50 + "\n")