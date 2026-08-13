"""
Step 4: PyTorch Dataset class.
Loads the normalized CSVs from Step 2, runs preprocessing from Step 3,
applies augmentation (train only), and normalizes with ImageNet stats.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from preprocessing import preprocess_image

# ImageNet mean/std — required since ResNet-50 is pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """Augmentation used ONLY during training."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms():
    """No augmentation — used for validation/test (including IDRiD)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DRDataset(Dataset):
    def __init__(self, csv_path, transform=None, image_size=224):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        original_idx = idx
        attempts = 0
        max_attempts = 20  # if 20 images in a row all fail, something is systematically wrong

        while attempts < max_attempts:
            row = self.df.iloc[idx]
            image = preprocess_image(row["filepath"], size=self.image_size)

            if image is not None:
                if self.transform:
                    image = self.transform(image)
                else:
                    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                label = int(row["label"])
                return image, label

            # This image failed - try the next one instead
            idx = (idx + 1) % len(self.df)
            attempts += 1

        # If we get here, 20 consecutive images all failed - that's not
        # normal and means preprocessing thresholds or file paths are
        # wrong, not that these specific images are bad.
        raise RuntimeError(
            f"20 consecutive images failed preprocessing starting at index "
            f"{original_idx} (file: {self.df.iloc[original_idx]['filepath']}). "
            f"This usually means is_low_quality() thresholds are too strict, "
            f"or filepaths in the CSV are wrong. Check preprocessing.py."
        )


# ---------- Example usage ----------
if __name__ == "__main__":
    train_dataset = DRDataset("labels/aptos_labels.csv", transform=get_train_transforms())
    idrid_test_dataset = DRDataset("labels/idrid_test_labels.csv", transform=get_eval_transforms())

    print(f"Train size: {len(train_dataset)}")
    print(f"IDRiD held-out test size: {len(idrid_test_dataset)}")

    image, label = train_dataset[0]
    print(f"Sample image shape: {image.shape}, label: {label}")