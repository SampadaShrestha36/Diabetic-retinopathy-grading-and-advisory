"""
Step 6 (DenseNet-121 variant): Training loop.
Two-stage fine-tuning + weighted loss (for class imbalance) + QWK tracking.

Identical pipeline to train.py (ResNet-50), swapped to a DenseNet-121
backbone via model_densenet.py. Checkpoints/best-model files are saved
under *_densenet names so they don't collide with your ResNet-50 run.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

from dataset import DRDataset, get_train_transforms, get_eval_transforms
from model_densenet import build_model, freeze_backbone, unfreeze_backbone
from train import train_one_stage, evaluate, get_class_weights, device

if __name__ == "__main__":
    import sys
    USE_BEN_GRAHAM = "--ben_graham" in sys.argv

    # ---- Data ----
    # Combines APTOS + DDR into one pool for training, since both use the
    # same 5-class ICDR grading scale. Stratified 85/15 split for validation.
    import pandas as pd
    from sklearn.model_selection import train_test_split

    aptos_df = pd.read_csv("labels/aptos_labels.csv")
    ddr_df = pd.read_csv("labels/ddr_labels.csv")
    combined_df = pd.concat([aptos_df, ddr_df], ignore_index=True)

    print(f"APTOS: {len(aptos_df)} images | DDR: {len(ddr_df)} images | "
          f"Combined: {len(combined_df)} images")

    # Reuses the SAME split files as the ResNet-50 run (same random_state=42
    # and same source CSV), so both backbones are trained/evaluated on
    # identical train/val partitions for a fair comparison.
    train_df, val_df = train_test_split(
        combined_df, test_size=0.15, stratify=combined_df["label"], random_state=42
    )

    train_df.to_csv("labels/combined_train_split.csv", index=False)
    val_df.to_csv("labels/combined_val_split.csv", index=False)

    train_image_size = 224
    train_dataset = DRDataset("labels/combined_train_split.csv", transform=get_train_transforms(),
                               image_size=train_image_size, use_ben_graham=USE_BEN_GRAHAM)
    val_dataset = DRDataset("labels/combined_val_split.csv", transform=get_eval_transforms(),
                             image_size=train_image_size, use_ben_graham=USE_BEN_GRAHAM)

    # WeightedRandomSampler is OFF by default (see train.py's ResNet-50 run
    # notes: it hurt Mild-class accuracy there). Pass --sampler to re-enable
    # it if you want to test it again on this backbone.
    USE_SAMPLER = "--sampler" in sys.argv

    if USE_SAMPLER:
        from torch.utils.data import WeightedRandomSampler
        train_labels = train_df["label"].values
        class_sample_counts = np.array([len(np.where(train_labels == c)[0]) for c in range(5)])
        sample_weights = 1.0 / class_sample_counts[train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=2)
        print("Using WeightedRandomSampler")
    else:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
        print("Using plain shuffled sampling (no oversampling) - loss weighting only")

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # ---- Class weights for imbalance (computed on the training split only) ----
    class_weights = get_class_weights("labels/combined_train_split.csv").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Model ----
    model = build_model(num_classes=5, pretrained=True).to(device)

    # ---- Stage 1: frozen backbone, higher LR on head only ----
    model = freeze_backbone(model)
    optimizer_stage1 = torch.optim.Adam(model.classifier.parameters(), lr=1e-3, weight_decay=5e-4)
    model, _ = train_one_stage(model, train_loader, val_loader, optimizer_stage1,
                                criterion, num_epochs=5, stage_name="stage1_densenet")

    # ---- Stage 2: unfreeze deeper layers, lower LR across all trainable params ----
    # denseblock3 onward is DenseNet-121's equivalent of ResNet-50's default
    # "layer3 onward" - denseblock1/denseblock2 stay frozen as generic
    # low-level features, matching model.py's unfreeze_backbone default.
    model = unfreeze_backbone(model, unfreeze_from_layer="denseblock3")
    optimizer_stage2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=5e-4
    )
    model, best_qwk = train_one_stage(model, train_loader, val_loader, optimizer_stage2,
                                       criterion, num_epochs=60, stage_name="stage2_densenet", patience=8)

    print(f"Best validation QWK: {best_qwk:.4f}")