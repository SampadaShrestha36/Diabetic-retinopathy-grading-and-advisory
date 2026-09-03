"""
Evaluates the trained model on a held-out test/validation CSV and
reports Accuracy, Quadratic Weighted Kappa (QWK), and AUC.

Usage:
    python test.py                              # defaults to aptos_val_split.csv
    python test.py labels/idrid_test_labels.csv  # or point at any other labeled CSV
    python test.py --grad-cam-dir gradcam --grad-cam-limit 25
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score, confusion_matrix
import torch.nn.functional as F

from dataset import DRDataset, get_eval_transforms
from grad_cam import generate_gradcam
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(checkpoint_path="best_model_stage2.pt", num_classes=5):
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def run_evaluation(model, loader, num_classes=5):
    """
    Runs the model over every image in the loader and collects:
    - predicted class labels (for accuracy/QWK)
    - predicted class probabilities (for AUC)
    - true labels
    """
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_probs, num_classes=5):
    metrics = {}

    # ---- Accuracy: fraction of exactly correct predictions ----
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # ---- Quadratic Weighted Kappa: accounts for ordinal severity, ----
    # ---- penalizes far-off predictions more than near-miss ones ----
    metrics["qwk"] = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    # ---- AUC: multi-class one-vs-rest, averaged ----
    # Only computed if every class actually appears in y_true, otherwise
    # roc_auc_score fails on classes with zero true samples.
    present_classes = np.unique(y_true)
    if len(present_classes) >= 2:
        try:
            metrics["auc_macro"] = roc_auc_score(
                y_true, y_probs, multi_class="ovr", average="macro", labels=list(range(num_classes))
            )
            metrics["auc_weighted"] = roc_auc_score(
                y_true, y_probs, multi_class="ovr", average="weighted", labels=list(range(num_classes))
            )
        except ValueError as e:
            metrics["auc_macro"] = None
            metrics["auc_weighted"] = None
            print(f"AUC could not be computed: {e}")
    else:
        metrics["auc_macro"] = None
        metrics["auc_weighted"] = None
        print("AUC skipped: fewer than 2 classes present in this test set")

    return metrics


def print_report(metrics, y_true, y_pred):
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {metrics['qwk']:.4f}")
    if metrics["auc_macro"] is not None:
        print(f"AUC (macro avg)     : {metrics['auc_macro']:.4f}")
        print(f"AUC (weighted avg)  : {metrics['auc_weighted']:.4f}")
    print("=" * 40)

    print("\nConfusion Matrix (rows = true grade, columns = predicted grade):")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    header = "      " + "".join([f"P{c:<6}" for c in range(5)])
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join([f"{v:<7}" for v in row])
        print(f"T{i}    {row_str}")


if __name__ == "__main__":
    # Defaults to the combined APTOS+DDR validation split. Point it at
    # labels/ddr_labels.csv or labels/aptos_labels.csv directly if you want
    # to check performance on one dataset in isolation, or at
    # labels/idrid_test_labels.csv once IDRiD is downloaded for a true
    # cross-population held-out test.
    test_csv = "labels/combined_val_split.csv"
    grad_cam_dir = None
    grad_cam_limit = None
    positional_args = []

    index = 1
    while index < len(sys.argv):
        argument = sys.argv[index]
        if argument == "--grad-cam-dir":
            grad_cam_dir = sys.argv[index + 1]
            index += 2
        elif argument == "--grad-cam-limit":
            grad_cam_limit = int(sys.argv[index + 1])
            index += 2
        else:
            positional_args.append(argument)
            index += 1

    if positional_args:
        test_csv = positional_args[0]
    checkpoint = "best_model_stage2.pt"

    print(f"Loading model from {checkpoint}...")
    model = load_trained_model(checkpoint)

    print(f"Loading test data from {test_csv}...")
    test_dataset = DRDataset(test_csv, transform=get_eval_transforms())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"Running inference on {len(test_dataset)} images...")
    y_true, y_pred, y_probs = run_evaluation(model, test_loader)

    metrics = compute_metrics(y_true, y_pred, y_probs)
    print_report(metrics, y_true, y_pred)

    if grad_cam_dir:
        print(f"Generating Grad-CAM explanations in {grad_cam_dir}...")
        generated = generate_gradcam(
            model,
            test_dataset.df,
            get_eval_transforms(),
            device,
            grad_cam_dir,
            limit=grad_cam_limit,
        )
        print(f"Generated {generated} Grad-CAM explanation(s).")