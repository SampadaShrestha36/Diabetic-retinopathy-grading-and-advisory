"""
Step 3: Preprocessing functions for fundus images.
Each function does ONE job, so you can test them individually before combining.
"""

import cv2
import numpy as np


def circular_crop(image):
    """
    Removes the black border around the retina by finding the actual
    eye region and cropping tightly around it.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image  # fallback: no crop if detection fails

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = image[y:y + h, x:x + w]
    return cropped


def apply_clahe(image):
    """
    Enhances contrast so lesions (microaneurysms, hemorrhages) become
    more visible. Applied on the green channel, which carries the most
    lesion detail in fundus images.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a, b))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    return enhanced_rgb


def resize_image(image, size=224):
    """Resizes to a fixed size expected by ResNet-50."""
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)


def is_low_quality(image, blur_threshold=5.0, dark_threshold=20.0):
    """
    Flags an image as low quality if it's too blurry or too dark to be
    gradable. Returns True if the image should be EXCLUDED.

    blur_threshold was recalibrated based on the actual APTOS 2019 blur
    score distribution (Laplacian variance on cropped images): min 3.26,
    5th percentile 5.91, median 20.36, max 113.03. Fundus images are
    naturally soft compared to typical sharp photos, so a generic
    threshold of 100 was excluding almost the entire dataset. 5.0 sits
    just below the 5th percentile, filtering only genuine outliers near
    the bottom of the range instead of the normal spread of the data.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur check: low variance in Laplacian = blurry image
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < blur_threshold:
        return True

    # Darkness check: very low average brightness = underexposed
    mean_brightness = gray.mean()
    if mean_brightness < dark_threshold:
        return True

    return False


def preprocess_image(image_path, size=224):
    """
    Full pipeline for one image: load -> crop -> quality check -> CLAHE -> resize.
    Cropping happens BEFORE the quality check, because the black border
    around the retina has near-zero pixel variance and falsely triggers
    the blur check if checked first. Quality must be judged on the actual
    retina, not the padding around it.
    Returns None if the image fails the quality check (so it gets skipped).
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = circular_crop(image)

    if is_low_quality(image):
        return None

    image = apply_clahe(image)
    image = resize_image(image, size=size)

    return image