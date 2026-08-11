"""
Step 6: Training loop.
Two-stage fine-tuning + weighted loss (for class imbalance) + QWK tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

from dataset import DRDataset, get_train_transforms, get_eval_transforms
from model import build_model, freeze_backbone, unfreeze_backbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_weights(csv_path):
    """
    Computes weights inversely proportional to class frequency, so rare
    classes (e.g. Severe, Proliferative DR) count more during training.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    labels = df["label"].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader):
    """Runs the model on a validation/test set and computes QWK."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    return qwk


def train_one_stage(model, train_loader, val_loader, optimizer, criterion,
                     num_epochs, stage_name, patience=3):
    """Trains for a number of epochs, stopping early if val QWK stops improving."""
    best_qwk = -1
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_qwk = evaluate(model, val_loader)
        print(f"[{stage_name}] Epoch {epoch+1}/{num_epochs} "
              f"- loss: {running_loss/len(train_loader):.4f} - val QWK: {val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"best_model_{stage_name}.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    return model, best_qwk


if __name__ == "__main__":
    # ---- Data ----
    train_dataset = DRDataset("labels/aptos_labels.csv", transform=get_train_transforms())
    val_dataset = DRDataset("labels/ddr_val_labels.csv", transform=get_eval_transforms())
    idrid_test_dataset = DRDataset("labels/idrid_test_labels.csv", transform=get_eval_transforms())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    idrid_test_loader = DataLoader(idrid_test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # ---- Class weights for imbalance ----
    class_weights = get_class_weights("labels/aptos_labels.csv").to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Model ----
    model = build_model(num_classes=5, pretrained=True).to(device)

    # ---- Stage 1: frozen backbone, higher LR on head only ----
    model = freeze_backbone(model)
    optimizer_stage1 = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    model, _ = train_one_stage(model, train_loader, val_loader, optimizer_stage1,
                                criterion, num_epochs=5, stage_name="stage1")

    # ---- Stage 2: unfreeze deeper layers, lower LR across all trainable params ----
    model = unfreeze_backbone(model, unfreeze_from_layer="layer3")
    optimizer_stage2 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5
    )
    model, best_qwk = train_one_stage(model, train_loader, val_loader, optimizer_stage2,
                                       criterion, num_epochs=15, stage_name="stage2")

    print(f"Best validation QWK: {best_qwk:.4f}")

    # ---- Final check: generalization to South Asian population (IDRiD) ----
    idrid_qwk = evaluate(model, idrid_test_loader)
    print(f"IDRiD held-out test QWK: {idrid_qwk:.4f}")