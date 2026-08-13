"""
Diagnostic: checks a sample of APTOS images against is_low_quality()
to see how many are being (possibly wrongly) flagged as bad, and why.
"""

import pandas as pd
import cv2
import numpy as np
from preprocessing import circular_crop, is_low_quality

df = pd.read_csv("labels/aptos_train_split.csv")

sample = df.sample(n=min(200, len(df)), random_state=42)

fail_quality = 0
fail_load = 0
passed = 0

for _, row in sample.iterrows():
    image = cv2.imread(row["filepath"])
    if image is None:
        fail_load += 1
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = circular_crop(image_rgb)

    if is_low_quality(image_rgb):
        fail_quality += 1
    else:
        passed += 1

print(f"Sample size: {len(sample)}")
print(f"Failed to load (bad filepath): {fail_load}")
print(f"Failed quality check: {fail_quality}")
print(f"Passed: {passed}")
print(f"Pass rate: {passed / len(sample) * 100:.1f}%")