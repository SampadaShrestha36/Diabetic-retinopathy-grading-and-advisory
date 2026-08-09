"""
Step 2: Normalize APTOS 2019, DDR, and IDRiD label files into one common format.
Output: a single CSV per dataset with columns [filepath, label]
This makes every dataset look identical to the code that loads them later.
"""

import pandas as pd
import os

# ---------- APTOS 2019 ----------
def prepare_aptos(csv_path, image_dir, out_csv):
    df = pd.read_csv(csv_path)  # columns: id_code, diagnosis
    df["filepath"] = df["id_code"].apply(lambda x: os.path.join(image_dir, f"{x}.png"))
    df["label"] = df["diagnosis"]

    # Drop rows where the image file doesn't actually exist on disk
    # (e.g. files that failed to extract due to a corrupted zip)
    before = len(df)
    df = df[df["filepath"].apply(os.path.exists)]
    dropped = before - len(df)
    if dropped > 0:
        print(f"WARNING: {dropped} images missing/corrupted and skipped")

    df[["filepath", "label"]].to_csv(out_csv, index=False)
    print(f"APTOS: {len(df)} images written to {out_csv}")


# # ---------- DDR ----------
# def prepare_ddr(txt_path, image_dir, out_csv):
#     # DDR label files are usually space-separated: filename label
#     df = pd.read_csv(txt_path, sep=" ", header=None, names=["filename", "label"])
#     df["filepath"] = df["filename"].apply(lambda x: os.path.join(image_dir, x))
#     df[["filepath", "label"]].to_csv(out_csv, index=False)
#     print(f"DDR: {len(df)} images written to {out_csv}")


# # ---------- IDRiD ----------
# def prepare_idrid(csv_path, image_dir, out_csv):
#     df = pd.read_csv(csv_path)  # columns: Image name, Retinopathy grade, Risk of macular edema
#     df["filepath"] = df["Image name"].apply(lambda x: os.path.join(image_dir, f"{x}.jpg"))
#     df["label"] = df["Retinopathy grade"]
#     df[["filepath", "label"]].to_csv(out_csv, index=False)
#     print(f"IDRiD: {len(df)} images written to {out_csv}")


if __name__ == "__main__":
    # These paths assume the folder structure set up in Steps 1-4:
    # project_root/data/aptos2019/, project_root/data/ddr/, project_root/data/idrid/
    # Run `ls` inside each folder after extraction and fix any names that don't match.

    prepare_aptos(
        csv_path="data/aptos2019/train.csv",
        image_dir="data/aptos2019/train_images",
        out_csv="labels/aptos_labels.csv",
    )

    # # DDR: check the extracted folder - it's usually split into
    # # DDR-dataset/DR_grading/train, valid, test, each with its own txt label file
    # prepare_ddr(
    #     txt_path="data/ddr/DR_grading/train.txt",
    #     image_dir="data/ddr/DR_grading/train",
    #     out_csv="labels/ddr_train_labels.csv",
    # )
    # prepare_ddr(
    #     txt_path="data/ddr/DR_grading/valid.txt",
    #     image_dir="data/ddr/DR_grading/valid",
    #     out_csv="labels/ddr_val_labels.csv",
    # )

    # # IDRiD: keep train and test SEPARATE - test is your held-out generalization set
    # prepare_idrid(
    #     csv_path="data/idrid/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
    #     image_dir="data/idrid/B. Disease Grading/1. Original Images/a. Training Set",
    #     out_csv="labels/idrid_train_labels.csv",
    # )
    # prepare_idrid(
    #     csv_path="data/idrid/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    #     image_dir="data/idrid/B. Disease Grading/1. Original Images/b. Testing Set",
    #     out_csv="labels/idrid_test_labels.csv",
    # )