"""
Step 5 (EfficientNet-B0 variant): EfficientNet-B0 model, adapted for
5-class DR grading.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes=5, pretrained=True):
    """
    Loads EfficientNet-B0 pretrained on ImageNet, and replaces the final
    1000-class layer with a 5-class layer for DR grading (0-4).
    """
    model = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)

    # model.classifier is nn.Sequential(Dropout, Linear) by default.
    # model.classifier[1] is the final classification layer — swap the
    # whole classifier out for a single Linear, matching the ResNet/
    # DenseNet variants' head (model.fc / model.classifier there).
    num_features = model.classifier[1].in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model


def freeze_backbone(model):
    """
    Stage 1 of fine-tuning: lock every layer EXCEPT the new final layer.
    Only the classification head learns at first.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    return model


def unfreeze_backbone(model, unfreeze_from_layer="features.6"):
    """
    Stage 2 of fine-tuning: unlock the deeper layers so they can adapt
    to fundus images too, starting from `unfreeze_from_layer` onward.

    EfficientNet-B0's feature extractor is model.features, a Sequential
    of 9 blocks: features.0 (stem conv) -> features.1..7 (7 MBConv
    stages) -> features.8 (final 1x1 conv). Default "features.6"
    unfreezes features.6, features.7, features.8, and classifier —
    leaving features.0-5 (generic low-level features) frozen, mirroring
    ResNet-50/DenseNet-121's "roughly back half of the backbone" default.
    """
    unfreeze = False
    for name, param in model.named_parameters():
        if unfreeze_from_layer in name:
            unfreeze = True
        if unfreeze:
            param.requires_grad = True
    return model


# ---------- Example usage ----------
if __name__ == "__main__":
    model = build_model(num_classes=5, pretrained=True)

    # Stage 1: train only the new head
    model = freeze_backbone(model)
    print("Stage 1: backbone frozen, only classifier layer trainable")

    # ... train for a few epochs here ...

    # Stage 2: unfreeze deeper layers, continue training at a lower LR
    model = unfreeze_backbone(model, unfreeze_from_layer="features.6")
    print("Stage 2: features.6, features.7, features.8, and classifier are now trainable")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")