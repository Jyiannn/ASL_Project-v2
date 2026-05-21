import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

input_file = os.path.join(PROJECT_ROOT, "data", "static_asl_data.csv")
expected_fields = 64

healthy_rows = []
corrupted_count = 0

with open(input_file, mode='r', newline='') as f:
    reader = csv.reader(f)
    header = next(reader)
    healthy_rows.append(header)
    for i, row in enumerate(reader, start=2):
        if len(row) == expected_fields:
            healthy_rows.append(row)
        else:
            corrupted_count += 1
            print(f"Skipping corrupted line {i}: expected 64 fields, found {len(row)}")

with open(input_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(healthy_rows)

print(f"\n[DONE] Cleaned CSV file. Removed {corrupted_count} malformed rows.")