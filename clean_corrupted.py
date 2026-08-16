"""
Scans all images in a labels CSV, tries to actually decode each one,
and removes any that are genuinely corrupted (not just flagged as
low-quality) - so training never has to encounter them at all.
"""

import pandas as pd
import cv2
import sys

def clean_corrupted_images(csv_path, out_csv_path=None):
    if out_csv_path is None:
        out_csv_path = csv_path  # overwrite in place

    df = pd.read_csv(csv_path)
    good_rows = []
    corrupted = []

    for i, row in df.iterrows():
        image = cv2.imread(row["filepath"])
        if image is None:
            corrupted.append(row["filepath"])
        else:
            good_rows.append(row)

        if (i + 1) % 500 == 0:
            print(f"Checked {i + 1}/{len(df)}...")

    clean_df = pd.DataFrame(good_rows)
    clean_df.to_csv(out_csv_path, index=False)

    print(f"\nOriginal: {len(df)} images")
    print(f"Corrupted (removed): {len(corrupted)} images")
    print(f"Clean: {len(clean_df)} images written to {out_csv_path}")

    if corrupted:
        print("\nCorrupted files:")
        for f in corrupted:
            print(f"  {f}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "labels/combined_train_split.csv"
    clean_corrupted_images(csv_path)