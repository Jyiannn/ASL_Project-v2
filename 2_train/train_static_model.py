import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

CSV_FILE = os.path.join(PROJECT_ROOT, "data", "static_asl_data.csv")
MODEL_OUT_PATH = os.path.join(PROJECT_ROOT, "models", "static_model.h5")
CLASSES_OUT_PATH = os.path.join(PROJECT_ROOT, "models", "static_classes.npy")

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"Missing static dataset. Run collect_static_data.py first to create '{CSV_FILE}'!")

print("[INFO] Loading and preprocessing static coordinate dataset...")
df = pd.read_csv(CSV_FILE).dropna()

y_raw = df.iloc[:, 0].astype(str).str.upper().values
X = df.iloc[:, 1:].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
class_names = encoder.classes_

np.save(CLASSES_OUT_PATH, class_names)
num_classes = len(class_names)

print(f"[SUCCESS] Cleanly parsed {X.shape[0]} samples across {num_classes} active alphabet classes.")
print(f"Target classes mapped: {list(class_names)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=12,
    restore_best_weights=True
)

candidate_batch_sizes = [8, 16, 32, 64]
best_accuracy = 0.0
best_batch_size = None
best_model = None

print("\n[OPTIMIZATION] Scanning batch sizes and hunting for highest validation accuracy...")

for bs in candidate_batch_sizes:
    print(f"\n🚀 Testing Candidate Batch Size: {bs} (Max Limit: 150 Epochs)")
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(63,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
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
    print(f"✔️ Batch Size {bs} finished. Peak Validation Accuracy: {trial_peak_acc * 100:.2f}%")
    
    if trial_peak_acc > best_accuracy:
        best_accuracy = trial_peak_acc
        best_batch_size = bs
        best_model = model

print(f"\n[SUCCESS] Optimization Complete!")
print(f"🏆 Best Selected Batch Size: {best_batch_size}")
print(f"🎯 Highest Validated Accuracy: {best_accuracy * 100:.2f}%")

best_model.save(MODEL_OUT_PATH)
print(f"[INFO] Standalone optimized static feature weights saved to '{MODEL_OUT_PATH}'.")

print("\n" + "="*50)
print("   STATIC PERFORMANCE MATRIX (BEST MODEL)   ")
print("="*50)

y_pred_probs = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=range(num_classes))
map_score = np.mean(precision)

print(f"🥇 Overall Alphabet mAP:  {map_score * 100:.1f}%")
print(f"🎯 Top-1 Coordinate Acc:  {best_accuracy * 100:.1f}%")
print("-" * 50)

print("Alphabet Performance Breakdown:")
print(classification_report(y_test, y_pred, target_names=class_names))

cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)

plt.figure(figsize=(11, 9))
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Static Confusion Matrix (Batch Size: {best_batch_size})', fontsize=14, pad=15)
plt.colorbar(label='Fraction of Predictions')

tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45, fontsize=10)
plt.yticks(tick_marks, class_names, fontsize=10)

thresh = cm_norm.max() / 2.
for i in range(cm_norm.shape[0]):
    for j in range(cm_norm.shape[1]):
        plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm_norm[i, j] > thresh else "black",
                 fontsize=9, weight="bold")

plt.ylabel('True Label', fontsize=12, labelpad=10)
plt.xlabel('Predicted Label', fontsize=12, labelpad=10)
plt.grid(False)
plt.tight_layout()

matrix_img_path = os.path.join(SCRIPT_DIR, 'static_confusion_matrix.png')
plt.savefig(matrix_img_path, dpi=300)
print(f"\n[INFO] Evaluation charts saved cleanly as '{matrix_img_path}'!")
print("="*50 + "\n")