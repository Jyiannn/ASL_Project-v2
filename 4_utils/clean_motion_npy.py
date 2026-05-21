import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_FILE = os.path.join(PROJECT_ROOT, "data", "motion_asl_data.npy")

if not os.path.exists(DATA_FILE):
    print("No motion data file found to clean.")
    exit()

with open(DATA_FILE, 'rb') as f:
    X = np.load(f)
    y = np.load(f)

print(f"Current Shape: X={X.shape}, y={y.shape}")

TARGET_INDEX_TO_DELETE = 0 

if TARGET_INDEX_TO_DELETE >= len(y):
    print("[ERROR] Index out of range.")
    exit()

keep_mask = np.ones(len(y), dtype=bool)
keep_mask[TARGET_INDEX_TO_DELETE] = False

X_clean = X[keep_mask]
y_clean = y[keep_mask]

with open(DATA_FILE, 'wb') as f:
    np.save(f, X_clean)
    np.save(f, y_clean)

print(f"[SUCCESS] Dropped entry {TARGET_INDEX_TO_DELETE}. New sample count: {len(y_clean)}")