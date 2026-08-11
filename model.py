"""
Step 5: ResNet-50 model, adapted for 5-class DR grading.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes=5, pretrained=True):
    """
    Loads ResNet-50 pretrained on ImageNet, and replaces the final
    1000-class layer with a 5-class layer for DR grading (0-4).
    """
    model = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)

    # model.fc is the final classification layer — swap it out
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def freeze_backbone(model):
    """
    Stage 1 of fine-tuning: lock every layer EXCEPT the new final layer.
    Only the classification head learns at first.
    """
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    return model


def unfreeze_backbone(model, unfreeze_from_layer="layer3"):
    """
    Stage 2 of fine-tuning: unlock the deeper layers so they can adapt
    to fundus images too, starting from `unfreeze_from_layer` onward
    (layer3, layer4, and fc — leaving early layers like layer1/layer2 frozen,
    since those hold generic low-level features that don't need retraining).
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
    print("Stage 1: backbone frozen, only fc layer trainable")

    # ... train for a few epochs here ...

    # Stage 2: unfreeze deeper layers, continue training at a lower LR
    model = unfreeze_backbone(model, unfreeze_from_layer="layer3")
    print("Stage 2: layer3, layer4, and fc are now trainable")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")