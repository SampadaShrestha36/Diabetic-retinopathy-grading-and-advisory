"""
Evaluates the trained DenseNet-121 model on a held-out test/validation CSV
and reports Accuracy, Quadratic Weighted Kappa (QWK), and AUC.

Usage:
    python test_densenet.py                              # defaults to combined_val_split.csv
    python test_densenet.py labels/ddr_labels.csv         # or point at any other labeled CSV
    python test_densenet.py --grad-cam-dir gradcam --grad-cam-limit 25
"""

import sys
import torch
from torch.utils.data import DataLoader

from dataset import DRDataset, get_eval_transforms
from grad_cam import generate_gradcam
from model_densenet import build_model
from test import run_evaluation, compute_metrics, print_report, device


def load_trained_model(checkpoint_path="best_model_stage2_densenet.pt", num_classes=5):
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    # Defaults to the combined APTOS+DDR validation split. Point it at
    # labels/ddr_labels.csv or labels/aptos_labels.csv directly if you want
    # to check performance on one dataset in isolation.
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
    checkpoint = "best_model_stage2_densenet.pt"

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